
import joblib
from sklearn.metrics import classification_report

import util
from extractfeatures import ExtractFeatures


class Classification:

    def __init__(self, model, features, labels):
        self.model = joblib.load(model)
        self.x = features
        self.y = labels

    def classifier(self):
        y_pred = self.model.predict(self.x)
        return classification_report(self.y, y_pred)


if __name__ == '__main__':

    # algorithms = ['bayes.pkl', 'lre.pkl', 'nn.pkl', 'svm.pkl', 'tree.pkl']
    algorithms = ['svm.pkl']
    # model = 'trained-model/assin_res_mod_skip300'
    model = 'trained-model/assin+msr_mod_glove50'
    # model = 'trained-model/mod_'
    # test = 'features/test/features-test-all1.txt'
    print('### Extracting features ###')
    features, _ = ExtractFeatures(model='model/glove50.txt', input_h='assin+msr/test/test-h.txt',
                                  input_t='assin+msr/test/test-t.txt').extract_features()
    # test = 'baseline/test/features-test.txt'
    labels = util.get_labels('assin+msr/labels-test.txt')
    print('### Classifying ###')
    for a in algorithms:
        print(Classification(model+a, features, labels).classifier())
