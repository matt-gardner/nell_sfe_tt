package edu.cmu.ml.rtw.pra

import edu.cmu.ml.rtw.pra.config.PraConfig
import edu.cmu.ml.rtw.pra.config.PraConfigBuilder
import edu.cmu.ml.rtw.pra.experiments.Dataset
import edu.cmu.ml.rtw.pra.experiments.Instance
import edu.cmu.ml.rtw.pra.experiments.Outputter
import edu.cmu.ml.rtw.pra.features.SubgraphFeatureGenerator
import edu.cmu.ml.rtw.pra.graphs.Graph
import edu.cmu.ml.rtw.pra.graphs.GraphBuilder
import edu.cmu.ml.rtw.pra.graphs.GraphInMemory
import edu.cmu.ml.rtw.pra.graphs.PprNegativeExampleSelector
import edu.cmu.ml.rtw.pra.models.BatchModel
import edu.cmu.ml.rtw.users.matt.util.FileUtil
import edu.cmu.ml.rtw.users.matt.util.JsonHelper
import edu.cmu.ml.rtw.users.matt.util.SpecFileReader

import edu.cmu.ml.rtw.tinkertoy.TinkerToy
import edu.cmu.ml.rtw.tinkertoy.TinkerToyServices

import scala.collection.mutable
import scala.collection.JavaConverters._
import scala.util.Random

import edu.cmu.ml.rtw.kb.KbManipulation
import edu.cmu.ml.rtw.kb.KbUtility
import edu.cmu.ml.rtw.kb.RTWValue
import edu.cmu.ml.rtw.shantytoy.TraditionalPredicateConsumer
import edu.cmu.ml.rtw.shantytoy.{SimpleInstance => ShantyToyInstance}

import org.json4s._
import org.json4s.JsonDSL._

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
  implicit val formats = DefaultFormats
  val random = new Random

  // SFE parameters
  val featureParams: JValue = params \ "features"
  val modelParams: JValue = params \ "learning"
  val maxTrainingInstances = JsonHelper.extractWithDefault(params, "max training instances", 5000)
  val maxPredictionInstances = JsonHelper.extractWithDefault(params, "max prediction instances", 1000)
  val svoFile = JsonHelper.extractWithDefault(params, "svo file", null: String)

  // Parameters for getting target relations from the KB
  val populateFilter = JsonHelper.extractWithDefault(params, "populate filter", true)
  val excludedRelations = JsonHelper.extractWithDefault(params, "relations to exclude",
    Seq("concept:latitudelongitude")).toSet
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
  val aliasEdge = "@ALIAS@"
  val excludedAliases = JsonHelper.extractWithDefault(params, "aliases to exclude", Seq[String]()).toSet

  val logDir = (params \ "log directory") match {
    case JNothing => null
    case JString(dir) => fileUtil.addDirectorySeparatorIfNecessary(dir)
    case _ => throw new RuntimeException("log directory parameter unrecognized")
  }

  def run() {
    println("Running NellSfeInterfacer")

    // Setting up the thing that will produce the proper NELL output.
    val tpc = new TraditionalPredicateConsumer()
    val properties = tts.getProperties()
    properties.setProperty("topK", Integer.valueOf(1000))
    properties.setProperty("revisedScoresFree", true)
    properties.setProperty("filterDerogatoryScores", true)
    tpc.initializeConsumer(tts, properties)

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
    //targetRelations.par.foreach(relation => {
    targetRelations.foreach(relation => runRelation(relation, graph, relationInstances,
      categoryInstances, inverses, domains, ranges, baseConfig, tpc))

    // Now that we have output from all of the relations, we create a final NELL TCF file.
    println("Outputting final NELL TCF file")
    tpc.writeTCF()
    println("Done running NellSfeInterfacer")
  }

  // A whole bunch of ugly parameters, but I needed to have this be its own method, so I could
  // return early if necessary.  Sorry...  This is one instance where a "continue" in scala would
  // make my code a lot nicer.  Maybe that means I designed it poorly...
  def runRelation(
      relation: RTWValue,
      graph: Graph,
      relationInstances: Map[RTWValue, Seq[Instance]],
      categoryInstances: Map[RTWValue, Set[Int]],
      inverses: Map[Int, Int],
      domains: Map[RTWValue, RTWValue],
      ranges: Map[RTWValue, RTWValue],
      baseConfig: PraConfig,
      tpc: TraditionalPredicateConsumer) {
    println(s"Running relation $relation")
    val trainingData = loadTrainingData(
      relation, graph, relationInstances, categoryInstances, domains, ranges)
    println(s"Training data size: ${trainingData.instances.size}")
    if (trainingData.instances.size == 0) {
      println("No training data found, so we're skipping this relation")
      return
    }
    val domain = categoryInstances(domains(relation))
    val range = categoryInstances(ranges(relation))
    if (domain.size > 400000) {
      println("WARNING: relation has more than 400000 concepts in domain; skipping...")
      return
    }

    // Relation-specific config stuff.
    val builder = new PraConfigBuilder(baseConfig)
    builder.setRelation(relation.asString())
    builder.setOutputBase(logDir + relation.asString() + "/")
    val relationIndex = graph.getEdgeIndex(relation.asString())
    val inverseIndex = inverses.getOrElse(relationIndex, -1)
    val unallowedEdges = if (inverseIndex == -1) Seq(relationIndex) else Seq(relationIndex, inverseIndex)
    builder.setUnallowedEdges(unallowedEdges)
    builder.setTrainingData(trainingData)
    val config = builder.build()
    fileUtil.mkdirs(config.outputBase)

    // Now we run the SFE code.

    val generator = new SubgraphFeatureGenerator(featureParams, config)
    val trainingMatrix = generator.createTrainingMatrix(trainingData)

    val model = BatchModel.create(modelParams, config)
    val featureNames = generator.getFeatureNames()
    model.train(trainingMatrix, trainingData, featureNames)

    val knownPositives = new Dataset(relationInstances(relation))
    // TODO(matt): try a simple BFS here, instead of a PPR computation
    val params: JValue = ("ppr computer" -> ("walks per source" -> 50) ~ ("log level" -> 4))
    val negativeExampleSelector = new PprNegativeExampleSelector(params, graph)

    val totalScores = new mutable.ArrayBuffer[(Instance, Double)]

    val numInstances = domain.size
    val notifyEvery = Math.min(10000, numInstances / 4)
    val done = new java.util.concurrent.atomic.AtomicInteger(0)
    println(s"Predicting new instances for the whole domain (${numInstances} concepts)")
    domain.par.foreach(concept => {
      val index = done.incrementAndGet()
      if (index % notifyEvery == 0) Outputter.info(s"Done with $index")
      val targets = negativeExampleSelector.findPotentialPredictions(concept, range, knownPositives)
      if (targets.size > 0) {
        val dataset = new Dataset(targets.map(target => Instance(concept, target, false, graph)).toSeq)
        val testMatrix = generator.createTestMatrix(dataset)
        val scores = model.classifyInstances(testMatrix)
        totalScores.synchronized {
          totalScores ++= scores
        }
      }
    })

    // And here we output the predictions in the format that NELL expects.  Well, we send them to
    // the TraditionalPredicateConsumer, which keeps track of them.
    outputPredictions(relation, totalScores, tpc)
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
    val start = compat.Platform.currentTime
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

    val end = compat.Platform.currentTime
    val seconds = (end - start) / 1000.0
    println(s"Done reading KB graph; took $seconds seconds")

    val categoryInstances = mutableCategoryInstances.map(entry => {
      (entry._1, entry._2.map(entityName => builder.nodeDict.getIndex(entityName)).toSet)
    }).withDefaultValue(Set[Int]()).toMap

    if (svoFile != null) {
      println(s"Adding SVO edges to graph from file $svoFile")
      println("Gathering a list of aliases first")
      val aliasStart = compat.Platform.currentTime
      val aliasMap = new mutable.HashMap[String, mutable.HashSet[String]]
      mutableCategoryInstances.foreach(entry => {
        entry._2.map(entity => {
          val strings = KbUtility.getLiteralStrings(KbManipulation.gpe(entity), null).asScala
          for (string <- strings) {
            aliasMap.getOrElseUpdate(string, new mutable.HashSet[String]).add(entity)
          }
        })
      })
      val aliases = aliasMap.keySet -- excludedAliases
      val aliasEnd = compat.Platform.currentTime
      val aliasSeconds = (aliasEnd - aliasStart) / 1000.0
      println(s"Took $aliasSeconds seconds")
      println("Now processing the SVO file")
      val count = new java.util.concurrent.atomic.AtomicInteger(0)
      val f: String => Seq[(String, String, String)] = (line: String) => {
        val index = count.getAndIncrement()
        fileUtil.logEvery(1000000, index, s"${index}; time so far: " +
          s"${(compat.Platform.currentTime - aliasEnd) / 1000.0}")
        val fields = line.split("\t")
        val s = fields(0)
        val v = fields(1)
        val o = fields(2)
        if (aliases.contains(s) && aliases.contains(o)) {
          Seq((s, v, o))
        } else {
          Seq[(String, String, String)]()
        }
      }
      val svoStart = compat.Platform.currentTime
      val edges = fileUtil.parFlatMapLinesFromFile(svoFile, f)
      val svoEnd = compat.Platform.currentTime
      val numEdges = edges.size  // doing this here to for computation, so the seconds is correct
      val svoSeconds = (svoEnd - svoStart) / 1000.0
      println(s"Found ${numEdges} SVO edges for the NELL entities in $svoSeconds seconds; adding them to the graph")
      val addedAliases = new mutable.HashSet[String]
      // TODO(matt): it'd be really nice if this were parallelizable...
      edges.foreach(edge => {
        val s = edge._1
        val v = edge._2
        val o = edge._3
        // builder.addEdge arguments are (source, target, relation)
        builder.addEdge(s, o, v)
        if (!addedAliases.contains(s)) {
          addedAliases.add(s)
          for (entity <- aliasMap(s)) {
            builder.addEdge(entity, s, aliasEdge)
          }
        }
        if (!addedAliases.contains(o)) {
          addedAliases.add(o)
          for (entity <- aliasMap(o)) {
            builder.addEdge(entity, o, aliasEdge)
          }
        }
      })
      // countAliases(edges)  // this is for seeing if there are any aliases that should be excluded
    }

    println(s"Graph has ${builder.edgesAdded} edges all together")
    val graph = builder.toGraphInMemory
    val relationInstances = mutableRelationInstances.map(entry => {
      (entry._1, entry._2.map(pair => {
        val arg1Index = graph.getNodeIndex(pair._1)
        val arg2Index = graph.getNodeIndex(pair._2)
        new Instance(arg1Index, arg2Index, true, graph)  // true here means this is a positive instance
      }).toSeq)
    }).withDefaultValue(Seq[Instance]()).toMap
    (graph, relationInstances, categoryInstances)
  }

  def countAliases(edges: Seq[(String, String, String)]) {
      val aliasCounts = new mutable.HashMap[String, Int]
      edges.foreach(edge => {
        val s = edge._1
        val v = edge._2
        val o = edge._3
        aliasCounts.update(s, aliasCounts.getOrElse(s, 0) + 1)
        aliasCounts.update(o, aliasCounts.getOrElse(o, 0) + 1)
      })
      val numToTake = 1000
      val topAliases = aliasCounts.map(entry => (-entry._2, entry._1)).toSeq.sorted.take(numToTake)
      println("Top aliases:")
      var totalCount = 0
      for (topAlias <- topAliases) {
        val count = topAlias._1
        val alias = topAlias._2
        println(s"${alias}: $count")
        totalCount += count
      }
      println(s"Total count in top ${numToTake}: $totalCount")
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
    // Note that we're downsampling here, because there may be too many training instances.  There
    // might be a better way to downsample, e.g., being sure to select seed instances, or using the
    // most confident instances.
    if (!relationInstances.contains(relation)) {
      println(s"Found no training instances for relation ${relation}!  Returning an empty dataset")
      return new Dataset(Seq())
    }
    val positiveDataset = new Dataset(random.shuffle(relationInstances(relation)).take(maxTrainingInstances))
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
    if (!relationInstances.contains(relation)) {
      println(s"Found no positive instances for relation ${relation}; skipping test data generation")
      return new Dataset(Seq())
    }
    val knownPositives = new Dataset(relationInstances(relation))
    val domain = categoryInstances(domains(relation))
    val range = categoryInstances(ranges(relation))
    // We'll stick with the default arguments to the negative example selector for now; they are
    // reasonable.
    val params: JValue = ("ppr computer" -> ("walks per source" -> 100))
    val negativeExampleSelector = new PprNegativeExampleSelector(params, graph)
    negativeExampleSelector.findPotentialPredictions(domain, range, knownPositives)
  }

  def outputPredictions(relation: RTWValue, scores: Seq[(Instance, Double)], tpc: TraditionalPredicateConsumer) {
    printTopScores(relation.asString(), scores, 10)
    scores.foreach(instanceScore => {
      val instance = instanceScore._1
      val prob = sigmoid(instanceScore._2)
      val arg1 = instance.graph.getNodeName(instance.source)
      val arg2 = instance.graph.getNodeName(instance.target)
      val shantyToyInstance = new ShantyToyInstance()
      shantyToyInstance.setCompoundProvenance("TODO")
      shantyToyInstance.setPredicate(relation)
      shantyToyInstance.setScore(prob)
      shantyToyInstance.setEnt1(KbManipulation.gpe(arg1))
      shantyToyInstance.setEnt2(KbManipulation.gpe(arg2))
      tpc.consumeInstance(shantyToyInstance)
    })
  }

  def sigmoid(x: Double): Double = {
    val exponent = Math.exp(-x)
    1 / (1 + exponent)
  }

  def printTopScores(relation: String, scores: Seq[(Instance, Double)], numToPrint: Int) {
    val topTen = scores.sortBy(pair => -pair._2).take(numToPrint)
    println(s"Top $numToPrint prediction for relation $relation:")
    for (prediction <- topTen) {
      val instance = prediction._1
      val score = prediction._2
      val arg1Name = instance.graph.getNodeName(instance.source)
      val arg2Name = instance.graph.getNodeName(instance.target)
      println(s"   $arg1Name, $arg2Name, $score")
    }
  }
}
