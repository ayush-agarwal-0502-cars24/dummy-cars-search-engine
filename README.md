# Dummy Cars Search Engine - 

By Ayush Agarwal 

# Vision of the project - 

We will need to make a freetext search engine for Buyer side on cars24.

Also I wanted to practice langchain.

# About the project - 


* Have taken a csv of cars dataset from kaggle (dataset originally from our competitor cardekho but was publically availaible)
* Made a vector db from the csv. Used FAISS.
* Made a search functionality - given query is searched using RAG (Retrieval Augmented Generation) and the related docs are returned.
* Made a streamlit dashboard to make this process interactive
* Have used tool calling in Langchain agents - Used Tavily API for searching web to get information related to the cars
* Have used memory functionality - When the user asks again (2nd query) then the agent remembers the older results, and then resuggests some models based on the older response plus user feedback (I could have added search and re-retrieval but too lazy)

# Future developments - 

* Have to add NER to detect color, company name, etc , in the input queries given by user.
* Have to study more about commmercial search engine for products design.
* I had a few ideas related to graphically improving UI/UX to ask the user for what kind of caar they want - for example - it is a binary choice that the person wants a 4-5 seater or a 6-7 seater ... and so on. This can help us in reducing the search space and deliver better recos.
* As the user continues to scroll, we can have pagination and then reranking models like lambdamart which can help in accounting for real time information, and give better recos.
* Have to add cars24 db to this

# Dry run - 

![image](https://github.com/user-attachments/assets/54da339b-7e40-4c2b-85a8-67321b72ad33)
