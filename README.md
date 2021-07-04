#Python3.TP_Note

Programmation pour classifier automatiquement des documents.

##Première étape :
Le scipt récupère les pdf puis les convertis en image.  
Une fois en image, on applique une rotation si nécessaire et on applique pour chaque image une transformation en niveau de gris.  
Une fois effectué, on extrait de chaque image une liste de tokens.  


##Deuxième étapes :
Pour chaque liste de token associé à un document, on vient néttoyer la liste en retirant les stopwords, les chiffres ou encore les caractères spéciaux.  
On retourne ainsi, pour chaque document, une liste des tokens qui le compose. Cette liste servira de base pour la création de notre  bag of words.  

##Troisième étape :
On transforme chaque documment on un vecteur de taille n (nombre de mot différent sur l'ensemble des documents).  

##Quatrième étape :
Une fois les documents transformés à l'aide de la méthode bag of words, on mets en place une pipeline de test sur différent modèle de machine learning.  
On vient ainsi tenter de classifier les documents à l'aide des différents modèles : SVM, Naive Bayes et KNN  

Baribeaud Arthur  
BDIA3