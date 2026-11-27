import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
def evaluate_model(model,X,y,class_names):
    pred=np.argmax(model.predict(X,verbose=0),axis=1)
    return {"accuracy":float(accuracy_score(y,pred)),"report":classification_report(y,pred,labels=list(range(len(class_names))),target_names=class_names,output_dict=True,zero_division=0),"confusion_matrix":confusion_matrix(y,pred,labels=list(range(len(class_names))))}
