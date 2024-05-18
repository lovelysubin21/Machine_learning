import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_score,accuracy_score,recall_score,auc,confusion_matrix,classification_report
from sklearn.tree import plot_tree,DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sns
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px

st.set_page_config(page_title="Machine Learning Model",layout="wide")
st.title("Machine Learning Model")

#Creating option menu in the side bar
with st.sidebar:
    Selected=option_menu("Title",["EDA","Prediction","Evaluation Metrics","NLP Detailing","Image Processing","Customer Recomendation"],
                         menu_icon="menu-button-wide")

if Selected == "EDA":
    #step 1: load Csv File
    class SessionState:
        def __init__(self,**kwargs): # we can put mutliple no. in variables
            for key,val in kwargs.items():
                setattr(self,key,val)

#create an instance of SessionState
    session_state=SessionState(df=None)

    # Step 1: Load CSV File
    uploaded_file = st.file_uploader("Choose a CSV file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        session_state.df = df  # Save the data in the session state
    
    # Step 2: Display DataFrame
    if session_state.df is not None and st.button("Show DataFrame"):
        st.dataframe(session_state.df)
    

    # Drop Duplicates and NaN Values
    if session_state.df is not None:
        st.write("### Drop Duplicates and NaN Values")
        session_state.df = session_state.df.drop_duplicates()
        session_state.df = session_state.df.dropna()
        st.dataframe(session_state.df)
        st.success("Duplicates and NaN values dropped successfully!")
    
        
    if session_state.df is not None:
        st.write("### Summary Statistics")
        st.text(session_state.df.describe())


    #LabelEncoding 
    if session_state.df is not None:
        st.write("### Label Encoding")
        le=LabelEncoder()
        for col in session_state.df.columns:
            if session_state.df[col].dtype=="object" or session_state.df[col].dtype=="bool":
                session_state.df[col]=le.fit_transform(session_state.df[col])
        st.dataframe(session_state.df)
        st.success("Label Encoding completed Successfully")
    
    #One_Hot Encoding for categorical columns
    if session_state.df is not None:
        st.write("### One-Hot Encoding")
        categorical_columns=session_state.df.select_dtypes(include=["object"]).columns
        session_state.df=pd.get_dummies(session_state.df,columns=categorical_columns)
        st.dataframe(session_state.df)
        st.success("One-Hot Encoding completed Successfully")

    #datatime Format conversion
    if session_state.df is not None:
        st.write("### DateTime conversion")
        session_state.df["target_date"]=pd.to_datetime(session_state.df["target_date"])
        st.dataframe(session_state.df)
        st.success("DateTime format conversion completed successfully")

    #Plot Relationship Curve
    if session_state.df is not None:
        st.write("### Plot Relationship Curve")
        sampled_df=pd.DataFrame(session_state.df["avg_visit_time"].sample(min(100,len(session_state.df))))
        plt.figure(figsize=(12,8))
        sns.pairplot(sampled_df)
        st.pyplot()
        #Disable the warning about pyplotglobaluse
        st.set_option('deprecation.showPyplotGlobalUse',False)
    
    #Detect the Treat outliers
    if session_state.df is not None:
        st.write("### Detect and Treat Outliers")
        Q1=session_state.df["transactionRevenue"].quantile(0.25)
        Q3=session_state.df["transactionRevenue"].quantile(0.75)
        IQR=Q3-Q1
        session_state.df=session_state.df[~((session_state.df["transactionRevenue"]<(Q1 -1.5 *IQR))| (session_state.df["transactionRevenue"]>(Q3 + 1.5 * IQR)))]
        st.dataframe(session_state.df)
        st.success("Outliers detected and treated successfully")

    #Treat Skewness
    if session_state.df is not None:
        st.write("### Treat Skewness")
        session_state.df["latest_visit_number"] =np.log1p(session_state.df["latest_visit_number"])
        st.dataframe(session_state.df)
        st.success("Skewness treated Successfully")
    
    
    # Calculate Correlation and Plot Heatmap
    if session_state.df is not None:
        st.write("### Correlation Heatmap")
        correlation_matrix = session_state.df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
        st.pyplot()
        # Disable the warning about PyplotGlobalUse
        st.set_option('deprecation.showPyplotGlobalUse',False)

    # scatter plot matric 
    if session_state.df is not None:
        st.write("### Scatter Plot Matrix")
        sns.set(style="ticks", rc={"figure.autolayout": False})
        sampled_df = pd.DataFrame(session_state.df.sample(min(1000, len(session_state.df))))
        
        progress_bar = st.progress(0)
        for i in range(len(sampled_df.columns)):
            sns.pairplot(sampled_df, vars=[sampled_df.columns[i]], diag_kind='hist')
            progress_bar.progress((i + 1) / len(sampled_df.columns))
    
        st.pyplot()
        # Disable the warning about PyplotGlobalUse
        st.set_option('deprecation.showPyplotGlobalUse',False)

#Prediction model
if Selected=="Prediction":
    st.markdown("""
    <style>
        .st-ax {
                background-color: lightgreen;
        }

        .stTextInput input{
                background-color: lightgreen;
        }
                
        .stNumberInput input{
                background-color: lightgreen;
        }
        .stDateInput input{
                background-color: lightgreen;
        }
                
    </style>
    """,unsafe_allow_html=True)
    with open("model_rf.pkl", "rb") as mf:
        new_model = pickle.load(mf)

    # Input form
    with st.form("user_inputs"):
        with st.container():
            count_session = st.number_input("count_session")
            time_earliest_visit = st.number_input("time_earliest_visit")
            avg_visit_time = st.number_input("avg_visit_time")
            days_since_last_visit = st.number_input("days_since_last_visit")
            days_since_first_visit = st.number_input("days_since_first_visit")
            visits_per_day = st.number_input("visits_per_day")
            bounce_rate = st.number_input("bounce_rate")
            earliest_source = st.number_input("earliest_source")
            latest_source = st.number_input("latest_source")
            earliest_medium = st.number_input("earliest_medium")
            latest_medium = st.number_input("latest_medium")
            earliest_keyword = st.number_input("earliest_keyword")
            latest_keyword = st.number_input("latest_keyword")
            earliest_isTrueDirect = st.number_input("earliest_isTrueDirect")
            latest_isTrueDirect = st.number_input("latest_isTrueDirect")
            num_interactions = st.number_input("num_interactions")
            bounces = st.number_input("bounces")
            time_on_site = st.number_input("time_on_site")
            time_latest_visit = st.number_input("time_latest_visit")
    
        submit_button = st.form_submit_button(label="Submit")
    
    # Predict using the model
    if submit_button:
        test_data = np.array([
            [
                count_session, time_earliest_visit, avg_visit_time, days_since_last_visit, 
                days_since_first_visit, visits_per_day, bounce_rate, earliest_source, 
                latest_source, earliest_medium, latest_medium, earliest_keyword, 
                latest_keyword, earliest_isTrueDirect, latest_isTrueDirect, num_interactions, 
                bounces, time_on_site, time_latest_visit
            ]
        ])
        
        # Convert the data to float
        test_data = test_data.astype(float)
    
        # Make predictions
        predicted = new_model.predict(test_data)[0]
        prediction_proba = new_model.predict_proba(test_data)
    
        # Display the results
        st.write("Prediction:", predicted)
        st.write("Prediction Probability:", prediction_proba)

if Selected=="Evaluation Metrics":

    # Step 1: Load CSV File
    uploaded_file = st.file_uploader("Choose a CSV file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    
        # EDA and Preprocessing Steps
    
        # Duplicate Removal
        st.write("### Duplicate Removal")
        df1 = df.drop_duplicates()
        st.success("Duplicates removed successfully!")
    
        # NaN Value Fill
        st.write("### NaN Value Fill")
        df2 = df1.fillna(0)  # You can replace 0 with the desired value
        st.success("NaN values filled successfully!")
    
        # DateTime Format Conversion
        st.write("### DateTime Format Conversion")
        date_columns = df2.select_dtypes(include=['datetime']).columns
        for col in date_columns:
            df2[col] = pd.to_datetime(df2[col])
        st.success("DateTime Format Conversion completed successfully!")
    
        # Display DataFrame
        st.dataframe(df2)
    
        # Summary Statistics
        st.write("### Summary Statistics")
        st.write(df2.describe())
    
        # Feature Importance with Random Forest
        le = LabelEncoder()
        for col in df2.columns:
            if df2[col].dtype == 'object' or df2[col].dtype == 'bool':
                df2[col] = le.fit_transform(df2[col])
    
        X_train = df2.drop('has_converted', axis=1)
        y_train = df2['has_converted']
    
        # Plot feature importance
        st.write("### Feature Importance with Random Forest")
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        feature_importances = rf.feature_importances_
        
        feature_importance_df=pd.DataFrame({
            "Feature":X_train.columns,
            "Impotance":feature_importances
            })
        top_10_features=feature_importance_df.sort_values(by="Impotance",ascending=False).head(10)["Feature"].tolist()
        extra_feature="has_converted"
        df3 = df2[top_10_features + [extra_feature]]
        #columns=['count_session','time_earliest_visit','avg_visit_time','days_since_last_visit','days_since_first_visit','visits_per_day','bounce_rate','earliest_source','latest_source','earliest_medium','has_converted']
        
        # Streamlit code
        st.title('Top 10 Features Importance')
        st.bar_chart(top_10_features)
        
        st.set_option('deprecation.showPyplotGlobalUse', False)
    
        # Pie Chart using Feature Importance
        st.write("### Pie Chart using Feature Importance")
        fig, ax = plt.subplots()
        ax.pie(feature_importances, labels=feature_importances, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)
        
        # Random Forest Model Build
        #df3 = pd.DataFrame(df3)

        # Drop the 'has_converted' column
        X = df3.drop('has_converted', axis=1)
        y=df3['has_converted']
        

            
        # Random Forest Model Build
        model = RandomForestClassifier(n_estimators=150,random_state=40)
        
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        rf_predict=rf_model.predict(X_train)
        rf_accuracy = accuracy_score(y_train,rf_predict)
        rf_Precision=precision_score(y_train,rf_predict)
        rf_recall=recall_score(y_train,rf_predict)
        rf_f1=f1_score(y_train,rf_predict)
        
        # Display Random Forest Model results
        st.write("# Random Forest Model")
        st.write("Accuracy:", rf_accuracy)
        st.write("Precision:", rf_Precision)
        st.write("Recall:", rf_recall)
        st.write("F1_score:", rf_f1)
        
        # Decision Tree Model Build
        dt_model = DecisionTreeClassifier()
        dt_model.fit(X_train, y_train)
        dt_predict=dt_model.predict(X_train)
        dt_accuracy = accuracy_score(y_train,dt_predict)
        dt_Precision=precision_score(y_train,dt_predict)
        dt_recall=recall_score(y_train,dt_predict)
        dt_f1=f1_score(y_train,dt_predict)
        
        # Display Decision Tree Model results
        st.write("# Decision Tree Model")
        st.write("Accuracy:", dt_accuracy)
        st.write("Precision:", dt_Precision)
        st.write("Recall:", dt_recall)
        st.write("F1_score:", dt_f1)

    
        # KNN Model Build
        knn_model = KNeighborsClassifier()
        knn_model.fit(X_train, y_train)  # Use the X_train, y_train from the first block
        knn_predict=knn_model.predict(X_train)
        knn_accuracy = accuracy_score(y_train, knn_predict)
        knn_Precision=precision_score(y_train,knn_predict)
        knn_recall=recall_score(y_train,knn_predict)
        knn_f1=f1_score(y_train,knn_predict)
        
        
        # Display KNN Model results
        st.write("# KNN Model")
        st.write("Accuracy:", knn_accuracy)
        st.write("Precision:",knn_Precision)
        st.write("Recall:", knn_recall)
        st.write("F1_score:",knn_f1)
    
    # Display results in a table
        results_data = {
            'Model': ['Random Forest', 'Decision Tree', 'KNN'],
            'Accuracy': [rf_accuracy, dt_accuracy, knn_accuracy],
            'Precision': [rf_Precision, dt_Precision, knn_Precision],
            'Recall': [rf_recall, dt_recall, knn_recall],
            'F1_score': [rf_f1, dt_f1, knn_f1]
        }
        
        results_table = st.table(results_data)
            
    
        # Plotly Visualization
        fig = px.bar(
            x=['Random Forest', 'Decision Tree', 'KNN'],
            y=[rf_accuracy, dt_accuracy, knn_accuracy],
            labels={'y': 'Accuracy', 'x': 'Models'},
            title='Model Accuracy Comparison'
        )
        st.plotly_chart(fig)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
from spacy import displacy
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

if Selected=="NLP Detailing":

    #download NLTK resources
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("wordnet")

    #Download scapy model
    spacy.cli.download("en_core_web_sm")
    nlp=spacy.load("en_core_web_sm")

    text_data=[
        ("I love streamlit", "positive"),
        ("Streamlit is easy to use", "positive"),
        ("NLP processing in streamlit is great", "positive"),
        ("Streamlit helps in building interactive web apps", "positive"),
        ("I dislike bugs in streamlit", "negative"),
        ("Streamlit could improve in some areas", "negative"),
        ("NLP can be challenging for beginners", "negative"),
        ("I struggle with streamlit syntax", "negative")]


    df=pd.DataFrame(text_data,columns=["text","label"])

    X_train=df["text"]
    Y_train=df["label"]

    #streamlit app
    st.title("NLP Processing")

    #text input
    text_input=st.text_area("Enter text for NLP Processing")

    #Tokenization
    if st.checkbox("Tokenization"):
        tokens=word_tokenize(text_input)
        st.write("Tokens:",tokens)

    #stopword Removal 
    if st.checkbox("Stopword Removal"):
        stop_words=set(stopwords.words("english"))
        filtered_tokens=[word for word in tokens if word.lower() not in stop_words]
        st.write("Tokens after stopword removal:",filtered_tokens)

    #number removal
    if st.checkbox("Number removal"):
        filtered_tokens=[word for word in filtered_tokens if not word.isdigit()]
        st.write("Token after number removal:",filtered_tokens)

    # Special Character Removal 
    if st.checkbox("Special character Removal"):
        filter_tokens=[word for word in filtered_tokens if word.isalnum()]
        st.write("Token after special character removal:",filter_tokens)
    
    #Lemmatization
    if st.checkbox("Lemmatization"):
        Lemmatization=WordNetLemmatizer()
        Lemmatization_tokens=[Lemmatization.lemmatize(word) for word in filtered_tokens]
        st.write("Token after lemmatization:",Lemmatization_tokens)

    #Parts of Speech(POS)
    if st.checkbox("Parts of Speech (POS)"):
        doc=nlp(text_input)
        pos_tags=[(token.text,token.pos_) for token in doc]
        st.write("PArt of Speech:",pos_tags)
    
    #N-gram
    if st.checkbox("N-gram"):
        n=st.slider("Select N for N-gram",min_value=2,max_value=5,value=2,step=1)
        ngram_vectorizer=CountVectorizer(ngram_range=(n,n))
        X_gram=ngram_vectorizer.fit_transform([text_input])
        st.write(f"{n}-gram representation:",X_gram.toarray())

    # Text Classification
    if st.checkbox("Text Classification"):
        # Create a pipeline with CountVectorizer and MultinomialNB
        
            model = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('classifier', MultinomialNB())
            ])
        
            # Fit the model on the training data
            model.fit(X_train, Y_train)
        
            # Make predictions on the testing data
            Y_pred = model.predict(X_train)
        
            # Display evaluation metrics
            accuracy = accuracy_score(Y_train, Y_pred)
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write("Classification Report:\n", classification_report(Y_train, Y_pred))
        
    # Sentiment Analysis
    if st.checkbox("Sentiment Analysis"):
        model = Pipeline([
          ('vectorizer', CountVectorizer()),
          ('classifier', MultinomialNB())
          ])

        # Fit the model on the training data
        model.fit(X_train, Y_train)
        #Assuming binary sentiment classification (positive and negative)
        sentiment = "Positive" if model.predict([text_input])[0] == "positive" else "Negative"
        st.write(f"Sentiment: {sentiment}")

    # Word Cloud
    if st.checkbox("Word Cloud"):
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_input)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot()
    
    # Keyword Extraction
    if st.checkbox("Keyword Extraction"):
        keywords = nlp(text_input).ents
        st.write("Keywords:", [keyword.text for keyword in keywords])



from PIL import Image,ImageEnhance,ImageFilter
import re
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd 
import numpy as np
from scipy import ndimage
import easyocr

if Selected == "Image Processing":

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])  # Specify languages you want to recognize
    # Define function to perform OCR
    def perform_ocr(image):

      result = reader.readtext(image)
      extracted_text = ' '.join([entry[1] for entry in result])
      return extracted_text
          
    def convert_to_grayscale(image):
    # Convert the image to grayscale
        grayscale_image = image.convert("L")
        return grayscale_image
    
    def resize_image(image, new_width, new_height):
        # Resize the image with the specified width and height while maintaining the aspect ratio
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if new_width == 0:
            new_width = int(new_height * aspect_ratio)
        elif new_height == 0:
            new_height = int(new_width / aspect_ratio)

        resized_image = image.resize((new_width, new_height))
        return resized_image

    def rotate_image(image, angle):
        # Rotate the image by the specified angle in degrees
        rotated_image = image.rotate(angle)
        return rotated_image
    
    def crop_image(image, left, top, right, bottom):
        # Crop the image based on the specified coordinates
        cropped_image = image.crop((left, top, right, bottom))
        return cropped_image
    
    def create_mirror_image(image):
        # Create a mirror image by flipping horizontally
        mirror_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return mirror_image
    
    def adjust_brightness(image, factor):
        # Adjust the brightness of the image
        enhancer = ImageEnhance.Brightness(image)
        brightened_image = enhancer.enhance(factor)
        return brightened_image

    def edge_detection(image):
        edge_detection=image.filter(ImageFilter.FIND_EDGES)
        edge_bright = ImageEnhance.Brightness(edge_detection)
        edge_ = edge_bright.enhance(9)
        return edge_

    def create_negative_image(image):
        # Convert the image to grayscale
        grayscale_image = image.convert("L")

        # Invert the colors to create a negative image
        negative_image = Image.fromarray(255 - np.array(grayscale_image))

        return negative_image


  
    def main():
      st.title("Image Processing")

      uploaded_file=st.file_uploader("Choose an image...",type=["jpg","jpeg","png"])

      if uploaded_file is not None:
          #display the upload image
          original_image=Image.open(uploaded_file)

          options=st.multiselect(
              "Choose Image Processing Steps",
              ["Preprocess","Grayscale","Resize","Rotate","Crop","Mirror","Brightness","Edge Detection","Negative_image"],
              default=["Grayscale"]
          )

          for option in options:
          #convert the image to grayscale
              if option == "Preprocess":
                    def main():
                        uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

                        if uploaded_image is not None:
                            st.sidebar.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

                            # Perform OCR
                            extracted_text = perform_ocr(uploaded_image.getvalue())

                            # Display extracted text
                            st.header("Extracted Text:")
                            st.write(extracted_text)

                    if __name__ == "__main__":
                      main()

              if option == "Grayscale":
                  grayscale_image=convert_to_grayscale(original_image)
                  st.image(grayscale_image,caption="Grayscale Image",use_column_width=True)

              if option == "Resize":
                  # Resize the image
                  new_width = st.slider("Select a width for resizing (leave 0 for auto):", 0, 1000, 300, step=10)
                  new_height = st.slider("Select a height for resizing (leave 0 for auto):", 0, 1000, 300, step=10)

                  resized_image = resize_image(original_image, new_width, new_height)
                  st.image(resized_image, caption=f"Resized Image ({new_width}x{new_height})", use_column_width=True)

              if option == "Rotate":
                  rotation_angle = st.slider("Select an angle for rotation:", -180, 180, 0, step=1)
                  rotated_image = rotate_image(original_image, rotation_angle)
                  st.image(rotated_image, caption=f"Rotated Image ({rotation_angle} degrees)", use_column_width=True)
              
              if option == "Crop":
                  st.write("Define crop coordinates:")
                  left = st.number_input("Left:", min_value=0, max_value=original_image.width, value=0)
                  top = st.number_input("Top:", min_value=0, max_value=original_image.height, value=0)
                  right = st.number_input("Right:", min_value=left, max_value=original_image.width, value=original_image.width)
                  bottom = st.number_input("Bottom:", min_value=top, max_value=original_image.height, value=original_image.height)
                  # Crop the image
                  cropped_image = crop_image(original_image, left, top, right, bottom)
                  st.image(cropped_image, caption="Cropped Image", use_column_width=True)

              if option == "Mirror":
                  mirror_image = create_mirror_image(original_image)
                  st.image(mirror_image, caption="Mirror Image", use_column_width=True)

              if option == "Brightness":
                  brightness_factor = st.slider("Adjust Brightness:", 0.1, 2.0, 1.0, step=0.1)
                  brightened_image = adjust_brightness(original_image, brightness_factor)
                  st.image(brightened_image, caption=f"Brightened Image (Factor: {brightness_factor})", use_column_width=True)

              if option == "Edge Detection":
                  edge=edge_detection(original_image)
                  st.image(edge,caption="Edge Detection",use_column_width=True)


              if option =="Negative_image":
                  negative_image = create_negative_image(original_image)
                  st.image(negative_image, caption="Negative Image", use_column_width=True)
              
  

    if __name__ =="__main__":
        main()





#Recommendation System

from surprise import SVD 
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split

if Selected =="Customer Recomendation":

    # Load CSV data
    def load_data():
        # Replace 'your_data.csv' with the actual path to your CSV file
        df = pd.read_csv('market_data.csv')
        return df

    # Create Surprise dataset
    def create_surprise_dataset(data):
        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(data[['CustomerId', 'Sub-Category_y','Rank']], reader)
        return dataset

    # Build collaborative filtering model
    def build_collaborative_filtering_model(dataset):
        trainset, testset = train_test_split(dataset, test_size=0.2)
        model = SVD()
        model.fit(trainset)
        return model, testset

    # Main function
    def main():
        st.title('Customer Recommendation System')

        # Load CSV data
        data = load_data()

        # Display original data
        st.write('### Original Data')
        st.write(data)

        # Create Surprise dataset
        dataset = create_surprise_dataset(data)

        # Build collaborative filtering model
        model, testset = build_collaborative_filtering_model(dataset)

        # Select a customer for recommendations
        selected_customer = st.selectbox('Select a CustomerId for recommendations:', data['CustomerId'].unique())

        # Generate recommendations
        if st.button('Generate Recommendations'):
            # Get unrated items for the selected customer
            unrated_items = data[data['CustomerId'] == selected_customer][['Sub-Category_y']]
            unrated_items = list(unrated_items['Sub-Category_y'])

            # Make predictions for unrated items
            predictions = [model.predict(selected_customer, item) for item in unrated_items]

            # Display top recommendations
            st.subheader(f'Top Recommendations for CustomerId {selected_customer}:')
            for prediction in sorted(predictions, key=lambda x: x.est, reverse=True)[:5]:
                st.write(f'Products: {prediction.iid}, Predicted Rating: {prediction.est:.2f}')

    if __name__ == '__main__':
        main()
