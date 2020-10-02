import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier



heart_data = pd.read_csv("data/heart_failure_clinical_records_dataset.csv")
#print(heart_data.head())


# fig = ff.create_distplot(hist_data, group_labels)
# fig.update_layout(title_text='Age Distribution plot')
#
# # fig.show()
#
# fig = px.box(heart_data, x='sex', y='age', points="all")
# fig.update_layout(
#     title_text="Gender wise Age Spread - Male = 1 Female =0")
#
# # fig.show()
#
# male = heart_data[heart_data["sex"] == 1]
# female = heart_data[heart_data["sex"] == 0]
#
# male_survivors = male[heart_data["DEATH_EVENT"] == 0]
# female_survivors = female[heart_data["DEATH_EVENT"] == 0]
# male_deaths = male[heart_data["DEATH_EVENT"] == 1]
# female_deaths = female[heart_data["DEATH_EVENT"] == 1]
#
# labels = ["Male-Survived", "Female-Survived", "Male-Deaths", "Female_Deaths"]
# values =  [len(male_survivors), len(female_survivors), len(male_deaths), len(female_deaths)]

# fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
# fig.update_layout(
#     title_text="Analysis on Survival - Gender")
#
# # fig.show()
#
# fig = px.violin(heart_data, y="age", x="sex", color="DEATH_EVENT", box=True, points="all", hover_data=heart_data.columns)
# fig.update_layout(title_text="Analysis in Age and Gender on Survival Status")
# fig.show()

#---------------------------------------

#let me save you the spoiler => ['time','ejection_fraction','serum_creatinine','age' ] => these are dominant features

Features = ['time','ejection_fraction','serum_creatinine','age']
x = heart_data[Features]
y = heart_data["DEATH_EVENT"]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=2698)

accuracy_list = []

r_clf = RandomForestClassifier(max_features=0.5, max_depth=15, random_state=1)
r_clf.fit(x_train, y_train)
r_pred = r_clf.predict(x_test)
r_acc = accuracy_score(y_test, r_pred)
accuracy_list.append(100*r_acc)



def predict(features, r_clf = r_clf):
    return {"live" : int(r_clf.predict([features]))}
