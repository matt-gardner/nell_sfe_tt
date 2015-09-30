package edu.cmu.ml.rtw.pra

import edu.cmu.ml.rtw.pra.config.JsonHelper
import edu.cmu.ml.rtw.pra.config.PraConfigBuilder
import edu.cmu.ml.rtw.pra.config.SpecFileReader
import edu.cmu.ml.rtw.pra.experiments.Dataset
import edu.cmu.ml.rtw.pra.experiments.Instance
import edu.cmu.ml.rtw.pra.experiments.Outputter
import edu.cmu.ml.rtw.pra.features.SubgraphFeatureGenerator
import edu.cmu.ml.rtw.pra.graphs.Graph
import edu.cmu.ml.rtw.pra.graphs.GraphBuilder
import edu.cmu.ml.rtw.pra.graphs.GraphInMemory
import edu.cmu.ml.rtw.pra.graphs.PprNegativeExampleSelector
import edu.cmu.ml.rtw.pra.models.PraModelCreator
import edu.cmu.ml.rtw.users.matt.util.FileUtil

import edu.cmu.ml.rtw.tinkertoy.TinkerToy
import edu.cmu.ml.rtw.tinkertoy.TinkerToyServices

import scala.collection.mutable
import scala.collection.JavaConverters._

import edu.cmu.ml.rtw.kb.KbManipulation
import edu.cmu.ml.rtw.kb.KbUtility
import edu.cmu.ml.rtw.kb.RTWValue

import org.json4s._

object SFETT extends TinkerToy {
  var params: JValue = null
  override def tinkerToyRun(tts: TinkerToyServices) {
    println("In tinkerToyRun")
    val interfacer = new NellSfeInterfacer(tts, params)
    interfacer.run()
    println("Finished with tinkerToyRun")
  }

  def main(args: Array[String]) {
    val paramFile = args(0)
    params = new SpecFileReader("").readSpecFile(paramFile)
    TinkerToyServices.runTinkerToy(this, args.drop(1));
    System.exit(0);
  }
}

class NellSfeInterfacer(tts: TinkerToyServices, params: JValue, fileUtil: FileUtil = new FileUtil) {
  println(params)

  implicit val formats = DefaultFormats

  // SFE parameters
  val featureParams: JValue = params \ "features"
  val modelParams: JValue = params \ "learning"

  // Parameters for getting target relations from the KB
  val populateFilter = JsonHelper.extractWithDefault(params, "populate filter", true)
  val excludedRelations = JsonHelper.extractWithDefault(params, "relations exclude",
    Seq("latitudelongitude")).toSet
  val targetRelations: Seq[RTWValue] = (params \ "relations") match {
    case relationsArray: JArray => {
      println(s"Found list of relations")
      relationsArray.children.map(jval => KbManipulation.gpe(jval.extract[String]))
    }
    case JNothing => {
      println("Getting target relations from KB")
      getTargetRelations()
    }
    case _ => throw new RuntimeException("relations parameter unrecognized")
  }
  println(s"Found ${targetRelations.size} relations")

  val logDir = (params \ "log directory") match {
    case JNothing => null
    case JString(dir) => fileUtil.addDirectorySeparatorIfNecessary(dir)
    case _ => throw new RuntimeException("log directory parameter unrecognized")
  }

  def run() {
    println("Running NellSfeInterfacer")

    // Base config stuff, for things that are consistent across relations, like the graph and the
    // inverse mapping.
    val baseBuilder = new PraConfigBuilder()
    baseBuilder.setOutputter(new Outputter(null))
    val (graph, relationInstances, categoryInstances) = loadGraph(logDir)
    baseBuilder.setGraph(graph)
    if (logDir != null) logGraph(logDir, graph, relationInstances, categoryInstances)
    val (inverses, domains, ranges) = getRelationMetadata(graph)
    baseBuilder.setRelationInverses(inverses)
    val baseConfig = baseBuilder.setNoChecks().build()

    // Now actually set up and run SFE for each target relation.
    targetRelations.par.foreach(relation => {
    //targetRelations.foreach(relation => {
      println(s"Running relation $relation")
      val trainingData = loadTrainingData(
        relation, graph, relationInstances, categoryInstances, domains, ranges)
      val testingData = generateTestData(
        relation, graph, relationInstances, categoryInstances, domains, ranges)

      // Relation-specific config stuff.
      val builder = new PraConfigBuilder(baseConfig)
      builder.setRelation(relation.asString())
      builder.setOutputBase(logDir + relation.asString() + "/")
      val relationIndex = graph.getEdgeIndex(relation.asString())
      val inverseIndex = inverses.getOrElse(relationIndex, -1)
      val unallowedEdges = if (inverseIndex == -1) Seq(relationIndex) else Seq(relationIndex, inverseIndex)
      builder.setUnallowedEdges(unallowedEdges)
      builder.setTrainingData(trainingData)
      builder.setTestingData(testingData)
      val config = builder.build()
      fileUtil.mkdirs(config.outputBase)

      // Now we run the SFE code.

      // The second parameter here is praBase, which we're not using (hopefully passing /dev/null works...)
      val generator = new SubgraphFeatureGenerator(featureParams, logDir, config)
      val trainingMatrix = generator.createTrainingMatrix(trainingData)

      val model = PraModelCreator.create(config, modelParams)
      val featureNames = generator.getFeatureNames()
      model.train(trainingMatrix, trainingData, featureNames)

      println(s"Scoring ${config.testingData.instances.size} potential predictions")
      val testMatrix = generator.createTestMatrix(config.testingData)
      val scores = model.classifyInstances(testMatrix)

      // And here we output the predictions in the format that NELL expects.
      outputPredictions(relation, scores)
    })
    println("Done running NellSfeInterfacer")
  }

  def getTargetRelations(): Seq[RTWValue] = {
    // TODO(matt): not sure this is done yet, but at least it's close.  I should talk to Bryan to
    // be sure.
    KbUtility.getConceptRelations(false).asScala.flatMap(relation => {
      if (excludedRelations.contains(relation.asString())) {
        Seq()
      } else if (populateFilter) {
        if (!KbManipulation.getValue("populate", relation).needString().equals("true")) {
          Seq()
        } else {
          Seq(relation)
        }
      } else {
        Seq(relation)
      }
    }).toSeq
  }

  def logGraph(
      logDir: String,
      graph: GraphInMemory,
      relationInstances: Map[RTWValue, Seq[Instance]],
      categoryInstances: Map[RTWValue, Set[Int]]) {
    // TODO(matt): the point of this is to allow for examining the KB graph when things go poorly.
    // Not sure it's totally worth the effort, but at least it might help diagnose some problems.
    // And, the logging methods really should be on the Graph object, so this will be a short
    // method when it's done (well, I'll have to log the relation and category instances, too...).
  }

  def graphIsLogged(logDir: String) = {
    fileUtil.fileExists(logDir + "graph/") &&
      fileUtil.fileExists(logDir + "relation_instances/") &&
      fileUtil.fileExists(logDir + "category_instances/")
  }

  def loadGraph(logDir: String): (GraphInMemory, Map[RTWValue, Seq[Instance]], Map[RTWValue, Set[Int]]) = {
    if (graphIsLogged(logDir)) {
      loadGraphFromLog(logDir)
    } else {
      loadGraphFromKb()
    }
  }

  def loadGraphFromLog(logDir: String): (GraphInMemory, Map[RTWValue, Seq[Instance]], Map[RTWValue, Set[Int]]) = {
    // TODO(matt): Once the logGraph method is actually written, we can load the graph from here,
    // to (hopefully) save time versus reading and traversing the KB tch file.  Until then, we'll
    // just load from the KB when this is called.
    loadGraphFromKb()
  }

  def loadGraphFromKb(): (GraphInMemory, Map[RTWValue, Seq[Instance]], Map[RTWValue, Set[Int]]) = {
    println("Building graph")
    val builder = new GraphBuilder
    // NOTE(matt): I based this method off of the old PRATT code, specifically
    // PRAGraph.addPromotedBeliefs.  That code had a bunch of stuff for Ni's PRA implementation
    // that is not necessary for my code, so I didn't just copy everything.  Hopefully I got all of
    // the important bits...

    println("Adding generalizations")
    val mutableCategoryInstances = new mutable.HashMap[RTWValue, mutable.ArrayBuffer[String]]

    // Not really sure what all of these parameters mean, but it's what I saw...
    val categoryIterator = KbUtility.getCategoryInstanceIterator(
      KbManipulation.gpe("concept:everypromotedthing"), true, false, true)
    while (categoryIterator.hasNext()) {
      val entity = categoryIterator.next()
      val entityName = entity.asString()
      for (category <- KbManipulation.getValue("generalizations", entity).iter().asScala) {
        val categoryName = category.asString()
        // builder.addEdge arguments are (source, target, relation)
        builder.addEdge(entityName, categoryName, "generalizations")
        addValueToArrayMap(mutableCategoryInstances, category, entityName)

        // We'll look for ancestor categories, but only for the purposes of keep track of which
        // instances belong to which categories, in case a relation has a domain or range that's
        // higher up in the hierarchy.
        for (ancestor <- KbUtility.getAncestorCategories(category).asScala) {
          addValueToArrayMap(mutableCategoryInstances, ancestor, entityName)
        }
      }
    }

    println(s"Found ${builder.edgesAdded} generalizations edges")
    println("Adding relations")
    val mutableRelationInstances = new mutable.HashMap[RTWValue, mutable.ArrayBuffer[(String, String)]]

    // Again, not really sure about this line, but I'm just copying here...
    val relationIterator =
      KbUtility.getRelationThreesomeIterator(KbManipulation.gpe("concept:relatedto"), true, false, true)
    while (relationIterator.hasNext()) {
      val triple = relationIterator.next()
      if (KbManipulation.isEntity(triple.arg2)) {
        val arg1Name = triple.arg1.asString()
        val arg2Name = triple.arg2.asString()
        val relationName = triple.relation.asString()
        // builder.addEdge arguments are (source, target, relation)
        builder.addEdge(arg1Name, arg2Name, relationName)
        addValueToArrayMap(mutableRelationInstances, triple.relation, (arg1Name, arg2Name))

        // I initially tried adding ancestor relations to the graph, but that turned out to be
        // somewhat catastrophic to running time.  If you're doing a BFS on the graph, adding an
        // additional 3ish edges per edge in the graph is pretty silly, and I don't think it's all
        // that informative for SFE, either.
      }
    }

    println("Done building graph")
    println(s"Graph has ${builder.edgesAdded} edges all together")
    val graph = builder.toGraphInMemory
    val categoryInstances = mutableCategoryInstances.map(entry => {
      (entry._1, entry._2.map(entityName => graph.getNodeIndex(entityName)).toSet)
    }).toMap
    val relationInstances = mutableRelationInstances.map(entry => {
      (entry._1, entry._2.map(pair => {
        val arg1Index = graph.getNodeIndex(pair._1)
        val arg2Index = graph.getNodeIndex(pair._2)
        new Instance(arg1Index, arg2Index, true, graph)  // true here means this is a positive instance
      }).toSeq)
    }).toMap
    (graph, relationInstances, categoryInstances)
  }

  def addValueToArrayMap[T](
      map: mutable.HashMap[RTWValue, mutable.ArrayBuffer[T]],
      key: RTWValue,
      value: T) {
    if (!map.contains(key)) {
        map(key) = new mutable.ArrayBuffer[T](100)
    }
    map(key) += value
  }

  def getRelationMetadata(graph: Graph): (Map[Int, Int], Map[RTWValue, RTWValue], Map[RTWValue, RTWValue]) = {
    val inverses = new mutable.HashMap[Int, Int]
    val domains = new mutable.HashMap[RTWValue, RTWValue]
    val ranges = new mutable.HashMap[RTWValue, RTWValue]
    for (relation <- KbUtility.getConceptRelations(false).asScala) {
      val relationIndex = graph.getEdgeIndex(relation.asString())
      val inverse = KbManipulation.getValue("inverse", relation).needScalar()
      val inverseIndex = graph.getEdgeIndex(inverse.asString())
      inverses(relationIndex) = inverseIndex
      inverses(inverseIndex) = relationIndex
      domains(relation) = KbManipulation.getValue("domain", relation).needScalar()
      ranges(relation) = KbManipulation.getValue("range", relation).needScalar()
    }
    (inverses.toMap, domains.toMap, ranges.toMap)
  }

  def loadTrainingData(
      relation: RTWValue,
      graph: Graph,
      relationInstances: Map[RTWValue, Seq[Instance]],
      categoryInstances: Map[RTWValue, Set[Int]],
      domains: Map[RTWValue, RTWValue],
      ranges: Map[RTWValue, RTWValue]): Dataset = {
    // NOTE(matt): we may need to downsample here, if it turns out that using all of the training
    // data is too expensive.  Let's check running times on a real KB first, though.
    val positiveDataset = new Dataset(relationInstances(relation))
    val allowedSources = categoryInstances(domains(relation))
    val allowedTargets = categoryInstances(ranges(relation))

    // We'll stick with the default arguments to the negative example selector for now; they are
    // reasonable.
    val negativeExampleSelector = new PprNegativeExampleSelector(JNothing, graph)
    val negativeExamples = negativeExampleSelector.selectNegativeExamples(
      positiveDataset, allowedSources, allowedTargets)
    positiveDataset.merge(negativeExamples)
  }

  def generateTestData(
      relation: RTWValue,
      graph: Graph,
      relationInstances: Map[RTWValue, Seq[Instance]],
      categoryInstances: Map[RTWValue, Set[Int]],
      domains: Map[RTWValue, RTWValue],
      ranges: Map[RTWValue, RTWValue]): Dataset = {
    val knownPositives = new Dataset(relationInstances(relation))
    val domain = categoryInstances(domains(relation))
    val range = categoryInstances(ranges(relation))
    // We'll stick with the default arguments to the negative example selector for now; they are
    // reasonable.
    val negativeExampleSelector = new PprNegativeExampleSelector(JNothing, graph)
    negativeExampleSelector.findPotentialPredictions(domain, range, knownPositives)
  }

  def outputPredictions(relation: RTWValue, scores: Seq[(Instance, Double)]) {
    val topTen = scores.sortBy(pair => -pair._2).take(10)
    println(s"Top ten prediction for relation ${relation.asString()}:")
    for (prediction <- topTen) {
      val instance = prediction._1
      val score = prediction._2
      val arg1Name = instance.graph.getNodeName(instance.source)
      val arg2Name = instance.graph.getNodeName(instance.target)
      println(s"   $arg1Name, $arg2Name, $score")
    }
    // TODO(matt): figure out what goes here
  }
}
