organization := "edu.cmu.ml.rtw"

name := "nell_sfe_tt"

scalaVersion := "2.11.2"

scalacOptions ++= Seq("-unchecked", "-deprecation")

javacOptions += "-Xlint:unchecked"

fork in run := true

javaOptions ++= Seq("-Xmx100g")

libraryDependencies ++= Seq(
  // NELL stuff that for some reason isn't included in OntologyLearner.jar
  "gnu.getopt" % "java-getopt" % "1.0.13",
  "com.cloudhopper" % "ch-tokyocabinet-java" % "1.24.0",
  "commons-collections" % "commons-collections" % "3.2.1",
  //  My dependencies
  "org.scalatest" % "scalatest_2.11" % "2.2.1" % "test",
  "edu.cmu.ml.rtw" %% "pra" % "3.1-SNAPSHOT"
)

instrumentSettings
