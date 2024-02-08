from flask import Flask, render_template,request
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from pathlib import Path
import sys

import networkx as nx

app = Flask(__name__, template_folder="./static/")

@app.route("/")
def websearch():
    return render_template("websearch.html")

@app.route("/imagesearch")
def imageserach():
      return render_template("imagesearch.html")  

@app.route("/reverseimagesearch")
def reverseimagesearch():
    return render_template("reverseimagesearch.html")


@app.route("/a")
def a():
    return render_template("A.html")

@app.route("/b")
def b():
    return render_template("B.html")
@app.route("/c")
def c():
    return render_template("C.html")
@app.route("/d")
def d():
    return render_template("D.html")
@app.route("/e")
def e():
    return render_template("E.html")

@app.route("/websearch", methods=['GET','POST'])
def web_search():
    if request.method == 'POST':
            query =request.form['query']
            if query == "":      
                return render_template("websearch.html")
         
            websites= ["http://127.0.0.1:5000/a",
                  "http://127.0.0.1:5000/b",
                  "http://127.0.0.1:5000/c",
                  "http://127.0.0.1:5000/d",
                  "http://127.0.0.1:5000/e"]
            
            tokenized_text = load_tokenized_text('tokenized_text_pickle.pkl')
          
            tfidf = TfidfVectorizer()
            tfidf_vectors = tfidf.fit_transform([' '.join(tokens) for tokens in tokenized_text])
            
            query_vector=tfidf.transform([query])
            similarities=cosine_similarity(query_vector,tfidf_vectors)
            
            if all_zeros(similarities[0]):
                return  render_template('notfound.html')
            #print(similarities)
            
            G = nx.DiGraph()
            
            for i, link in enumerate(websites):
                G.add_node(link)
                for j, sim in enumerate(similarities[0]):
                    
                    if sim>0 and i !=j:
                        G.add_edge(link,websites[j], weight=sim)
                    
            pagerank=nx.pagerank(G)
            ranked_result=sorted(pagerank.items(), key = lambda x: x[1], reverse=True) 
            top_results=[x[0] for x in ranked_result if x[1]>=0.14]
            #print(top_results)

            return render_template("results.html", data=[top_results,query])



@app.route("/search_images", methods=['GET','POST'])
def search_images():
    if request.method=='POST':
        query=request.form['query'].lower()
        
        if query=='':
            return render_template('imagesearch.html')
        
        with open('images.json','r') as f:
            images=json.load(f)
        
        results=[]
        for img in images:
            if query in img['alt_text'] or query in img['title']:
                 results.append(img)
            else:
                continue
                
        if len(results)==0:
            return render_template('notfound.html')
      #  print(results)
                    
        return render_template("imageresults.html", data=[results,query])
         
fe=FeatureExtractor()
features=[]
img_paths=[]

for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/reverse_img_store") / (feature_path.stem + ".jpg"))
features = np.array(features)     
            
@app.route("/reverseimagesearchresult", methods=['GET','POST']) 
def reverseimagesearchresult():
    if request.method=='POST':
        file=request.files['query_img']
        
        #save query image
        
        img=Image.open(file.stream) #PIL image
        uploaded_img_path='./static/uploaded/'+ \
                            datetime.now().isoformat().replace(":",'.')\
                            + "_" + file.filename
        img.save(uploaded_img_path)
        
        #run search
        query=fe.extract(img)
        print(query)
        
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        
        ids=np.argsort(dists)[:3]
        
        scores=[(dists[id],img_paths[id]) for id in ids]
        
        return render_template('reverseimagesearch.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    
 
    else:
        
        return render_template('reverseimagesearch.html')
        
    
                                 
    
def load_tokenized_text(filename):
    tokenized_text=pickle.load(open(filename,'rb'))
    return tokenized_text

def all_zeros(l):
    for i in l:
        if i!=0:
            return False
    return True
if __name__=='__main__':
    app.run(debug=True)
    
    
    
    
    
    
    
    
    
    
    