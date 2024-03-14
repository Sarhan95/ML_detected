########### importattion des packages necessaire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix



st.title("application de détection des frauds cartes bancaire")
st.subheader('auteur: Zakaria Sarhani')
st.markdown("Fraude bancaire : Détection proactive avec l'IA - Une application Streamlit")
st.image("depositphotos_16372601-stock-photo-business-woman-with-financial-symbols.jpg")

    #########importation des données
@st.cache_data(persist=True)
def laod_data():
    data=pd.read_csv("creditcard.csv")
    return data
        
        
df=laod_data()
df_sample=df.sample(100)
if st.sidebar.checkbox("afficher la base de données",False):
    st.subheader("jeu de donnes des cartes bancaire deja utilisé")
    st.write(df_sample)

    ######## inspection des données
if st.checkbox("afficher tableau valeur manquantes",False):
    data_manquantes=pd.DataFrame({
    'value_nombre':df.isnull().sum(),
    "proportion_manquantes":df.isna().sum()/df.shape[0]
    })
    data_manquantes.sort_values("proportion_manquantes",ascending=True)
    st.write("nombre valeur manquantes")
    st.write(data_manquantes)
    ######## division notre dataset
#@st.cache_data(persist=True)
def split(df):
    x=df.drop("Class",axis=1)
    y=df["Class"]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=123)
    st.write("dim de x_train:",x_train.shape)
    st.write("dim de y_train:",y_train.shape)
    st.write("dim de x_test:",x_test.shape)
    st.write("dim de y_test:",y_test.shape)
    return x_train,x_test,y_train,y_test
x_train,x_test,y_train,y_test=split(df)

######### deploiment des models###########"
####### creer un classificateur
classificateur=st.sidebar.selectbox("classificateur",
                                    ("random forest", "regression logistique","SVM")

) 
graphique_confusion = st.sidebar.checkbox(
        "affichage  graphique de matrix de confusion", False

    )
if classificateur=="regression logistique":
    st.sidebar.subheader("les hyperparamétre de rgression logistique")
    hyp_c = st.sidebar.number_input(
                "choisir la valeur de régularisation", 0.01, 10.0
            )
    n_max_itere = st.sidebar.number_input(
                "nombre d'itération ", 100, 1000, step=10)
    if st.sidebar.button("execution",key="classify"):
        st.subheader("resultatde regression logistique")

        model=LogisticRegression(
            random_state=123,
            max_iter=n_max_itere,
            C=hyp_c

        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
                ###### calculer les métrics
        accuracy = np.round(accuracy_score(y_test, y_pred), 2)
        recall = np.round(recall_score(y_test, y_pred), 2)
        precesion_score = np.round(precision_score(y_test, y_pred), 2)
        f_score = np.round(f1_score(y_test, y_pred), 2)

        st.write("accuracy:", accuracy)
        st.write("recall:", recall)
        st.write("precesion_score:", precesion_score)
        st.write("f_score:", f_score)

        cm = confusion_matrix(y_test, y_pred)
        if graphique_confusion:
            plt.figure(figsize=(6, 4))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=["transaction authentique", "transaction frauduleuse"],
                            yticklabels=["transaction authentique", "transaction frauduleuse"])
            plt.xlabel("Prédictions")
            plt.ylabel("Vraies valeurs")
            plt.title("Matrice de Confusion")
            st.pyplot()
            st.write("confusion matrix des class ")
            st.write(cm)


if st.checkbox("carte de graphique relation entre les variables quantitative",False):
    var_quant=df.select_dtypes(exclude="object").columns.to_list()
    x_var=st.selectbox("choisir le variableen abscisse",var_quant)   
    y_var=st.selectbox("choisir le variableen ordonnée",var_quant)

    fig2=px.scatter(data_frame=df,
            
                x=x_var,y=y_var,
                title=str(x_var)+"vs"+str(y_var)
                
               
    )
 
    st.plotly_chart(fig2) 


if classificateur=="random forest":
    st.sidebar.subheader("les hyperparamétres de model")
    criter=st.sidebar.radio("choisir le type de dévision",("gini","entropy"))
    n_arbre=st.sidebar.number_input("choisir le nombre arbres de decision ",100,1000,step=10)
    profondeur=st.sidebar.number_input("la profondeur de l'arbre",1,20,step=1)
    max_feature=st.sidebar.radio("choisir le nombre des charactirisque",("sqrt","log2"))
    if st.sidebar.button("execution",key="classify"):
        st.subheader("resultat de random forest")
        rd=RandomForestClassifier(random_state=123,max_features=max_feature,
                                  n_estimators= n_arbre,
                                  max_depth=profondeur,
                                  criterion=criter)
        rd.fit(x_train, y_train)
        y_pred = rd.predict(x_test)
                ###### calculer les métrics
        accuracy_rd = np.round(accuracy_score(y_test, y_pred), 2)
        recall_rd = np.round(recall_score(y_test, y_pred), 2)
        precesion_score_rd = np.round(precision_score(y_test, y_pred), 2)
        f_score_rd = np.round(f1_score(y_test, y_pred), 2)

        st.write("accuracy:", accuracy_rd)
        st.write("recall:", recall_rd)
        st.write("precesion_score:", precesion_score_rd)
        st.write("f_score:", f_score_rd)


        cm_confusion= confusion_matrix(y_test, y_pred)
        if graphique_confusion:
            plt.figure(figsize=(6, 4))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            sns.heatmap(cm_confusion, annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=["transaction authentique", "transaction frauduleuse"],
                            yticklabels=["transaction authentique", "transaction frauduleuse"])
            plt.xlabel("Prédictions")
            plt.ylabel("Vraies valeurs")
            plt.title("Matrice de Confusion")
            st.pyplot()
            st.write("confusion matrix des class ")
            st.write(cm_confusion)


if classificateur=="SVM":
    st.sidebar.subheader("les hyperparamétres de SVM")    
    c=st.sidebar.number_input("choix le paramétres de régularisation",0.01,10.0)
    krenel=st.sidebar.radio("coeficient pour le noyeau",("poly","rbf","sigmoid"))
    class_widght=st.sidebar.radio("poids specifier associe chaque class ",("balanced",None))
    n_max_tr=st.sidebar.number_input("nobmre itération",100,1000,step=10)
    if st.sidebar.button("execution",key="classify"):
        st.subheader("resultat de support vecteur machine")

        svm=SVC(random_state=123,
                C=c,
                kernel=krenel,
                class_weight=class_widght,
                max_iter=n_max_tr)
        svm.fit(x_train, y_train)
        y_pred = svm.predict(x_test)
                ###### calculer les métrics
        accuracy_svm = np.round(accuracy_score(y_test, y_pred), 2)
        recall_svm = np.round(recall_score(y_test, y_pred), 2)
        precesion_score_svm = np.round(precision_score(y_test, y_pred), 2)
        f_score_svm = np.round(f1_score(y_test, y_pred), 2)

        st.write("accuracy:", accuracy_svm)
        st.write("recall:", recall_svm)
        st.write("precesion_score:", precesion_score_svm)
        st.write("f_score:", f_score_svm)


        cm_confus= confusion_matrix(y_test, y_pred)
        if graphique_confusion:
            plt.figure(figsize=(6, 4))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            sns.heatmap(cm_confus, annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=["transaction authentique", "transaction frauduleuse"],
                            yticklabels=["transaction authentique", "transaction frauduleuse"])
            plt.xlabel("Prédictions")
            plt.ylabel("Vraies valeurs")
            plt.title("Matrice de Confusion")
            st.pyplot()
            st.write("confusion matrix des class ")
            st.write(cm_confus)


            

        



            






    

    




   

    
  
  
  
  
  
  
  