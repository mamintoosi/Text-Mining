import kindred
trainCorpus = kindred.bionlpst.load('2016-BB3-event-train')
devCorpus = kindred.bionlpst.load('2016-BB3-event-dev')
predictionCorpus = devCorpus.clone()
predictionCorpus.removeRelations()
classifier = kindred.RelationClassifier()
classifier.train(trainCorpus)
classifier.predict(predictionCorpus)
f1score = kindred.evaluate(devCorpus, predictionCorpus, metric='f1score')