#!/usr/bin/env python
# coding: utf-8

# In[101]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate  # For generating markdown tables
import mlflow
import os
from sklearn import metrics
os.environ["MLFLOW_TRACKING_URI"] = "https://gitlab-codecamp24.obss.io/api/v4/projects/129/ml/mlflow/"
os.environ["MLFLOW_TRACKING_TOKEN"] = "glpat-cxSaGfZ6sy-ifJaYJAmB"



df=pd.read_csv("tips.csv")
df.head()


# In[104]:


print(df.isna().sum())


# In[105]:


df['smoker'] = df['smoker'].map({'Yes': 1, 'No': 0})
df['sex'] = df['sex'].map({'Female': 1, 'Male': 0})
df['time'] = df['time'].map({'Dinner': 1, 'Lunch': 0})


# In[106]:


correlation = df['total_bill'].corr(df['tip'])
print('Pearson Korelasyon Katsayısı:', correlation)


# In[107]:


correlation = df['size'].corr(df['tip'])
print('Pearson Korelasyon Katsayısı:', correlation)


# In[108]:


day_mapping = {
    'Sun': 0,
    'Mon': 1,
    'Tue': 2,
    'Wed': 3,
    'Thur': 4,
    'Fri': 5,
    'Sat': 6
}

df['day'] = df['day'].map(day_mapping)


# In[109]:


sns.boxplot(x='sex', y='tip', data=df)
plt.title('Cinsiyete Göre tip Dağılımı')
plt.show()


# In[110]:


from sklearn.metrics.pairwise import cosine_similarity
cosine_sim_matrix = cosine_similarity(df.T)

cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=df.columns, columns=df.columns)

print(cosine_sim_df)


# In[111]:


df['combined_feature'] = df['size'] * df['total_bill']
df = df.drop(columns=['size'])
df = df.drop(columns=['total_bill'])


# In[112]:


Q1 = df['tip'].quantile(0.25)
Q3 = df['tip'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

mean_tip = df['tip'].mean()

df.loc[(df['tip'] < lower_bound) | (df['tip'] > upper_bound), 'tip'] = mean_tip

sns.boxplot(x='sex', y='tip', data=df)
plt.title('Cinsiyete Göre Tip Dağılımı')
plt.show()


# In[113]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
X = df.drop(columns=['tip'])
y = df['tip']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
model = ElasticNet(alpha=0.6, l1_ratio=1, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Elastic Net MSE: {mse}')


df['integer_part'] = df['tip'].apply(lambda x: int(x))
df['tip'] = df['integer_part']
df = df.drop(columns=['integer_part'])
y = df['tip']


# In[115]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[116]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[117]:


from sklearn.svm import SVC
model = SVC(kernel='linear') 
model.fit(X_train_scaled, y_train)


# In[118]:


from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")


# In[119]:


from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_accuracy


# In[120]:


from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_accuracy


# In[121]:


from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_accuracy


# Define Logistic Regression parameters
lr = LogisticRegression(random_state=RANDOM_SEED)

# Define DecisionTree parameters
dt = DecisionTreeClassifier(random_state=RANDOM_SEED)


mlflow.set_experiment("tip-prediction")
def eval_metrics(actual, pred):
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.mean_squared_error(actual, pred, pos_label=1)
    plt.figure(figsize=(8,8))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='lower right')
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/box_plot.png")
    # Close plot
    plt.close()
    return accuracy, f1

def save_model_report(metrics, params, name, report_path="metrics_report.md"):
    report = []
    if os.getenv('MLFLOW_TRACKING_URI'):
        if os.getenv('GITLAB_CI'):
            ci_job_id = os.getenv('CI_JOB_ID')
    else:
        ci_job_id = "Undefined"
    
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = f.readlines()
            
    report.append(f"\n\n# Model Report for {name}\n\n")
    report.append(f"#### CI Job ID: {ci_job_id}\n\n")
    report.append("## Model Parameters\n\n")
    for key, value in params.items():
        report.append(f"- **{key}** : {value}\n")
    report.append("\n\n## Metrics\n\n")

    if isinstance(metrics, dict):
        metrics_list = [metrics]
    else:
        metrics_list = metrics

    report.append(tabulate(metrics_list, headers="keys", tablefmt="pipe"))

    with open(report_path, "w") as f:
        f.write("".join(report))

    return report_path

def mlflow_logging(model, X, y, name):
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)
        if os.getenv('GITLAB_CI'):
            mlflow.set_tag('gitlab.CI_JOB_ID', os.getenv('CI_JOB_ID'))
        
        pred = model.predict(X)
        # Metrics
        accuracy, f1 = eval_metrics(y, pred)
        
        metrics_data = {
            "Mean CV score": model.best_score_,
            "Accuracy": accuracy,
            "f1-score": f1,
        }
        params = model.best_params_
        
        # Logging best parameters from GridSearchCV
        mlflow.log_params(params)
        mlflow.log_params({"Class": name})
        # Log the metrics
        mlflow.log_metrics(metrics_data)
        
        # Logging artifacts and model
        mlflow.log_artifact("plots/box_lot.png")
        
        # Save and log model report
        report_path = save_model_report(metrics_data, params, name)

        mlflow.sklearn.log_model(model, name) 

        mlflow.end_run()

mlflow_logging(model_tree, X_test, y_test, "DecisionTreeClassifier")
mlflow_logging(model_log, X_test, y_test, "LogisticRegression")




