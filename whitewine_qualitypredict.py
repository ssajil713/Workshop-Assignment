import joblib as jb
import numpy as np
model = jb.load('whitewine_model.keras')
result = model.predict(np.array([[7,0.27,0.36,20.7,0.045,45,170,1.001,3,0.45,8.8]]))
print("Predicted Quality=", result)