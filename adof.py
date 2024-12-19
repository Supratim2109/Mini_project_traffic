import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer


X = df.drop('Severity', axis=1)  
y = df['Severity']  
X = pd.get_dummies(X, columns=['Weather_Condition', 'Wind_Direction'], drop_first=True)
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(X.iloc[:, 2:9])
X.iloc[:, 2:9] = imputer.transform(X.iloc[:, 2:9])




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)




rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)



y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))
