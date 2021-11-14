import sys
import requests
from bs4 import BeautifulSoup
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import pathlib
from pathlib import Path
import csv
import collections
import re
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import nltk
import numpy as np
import hashlib
import pickle
import heapq

#style of the dataframe output
pd.set_option('display.max_colwidth', None)

"""
============================================================================================================

    html pages download functions

============================================================================================================

"""

def save_html_animePage(url, directoryNum, index):
    # For each page this function takes as input it saves the html of the anime in a folder.
    
    # Get current page
    req = requests.get(url)
    
    # MyAnimeList might stop the connection due to the numbers of request
    if(req.status_code != 200) : 
        raise Exception(f"Web site have closed the connection.\nRestart the process from page: {(index//50)}")
    
    # get the path where to place the file
    save_path = f"{pathlib.Path().resolve()}/animeList_pages/{directoryNum}th_page"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Write the file in the directory.
    if(sys.platform != "win32"):
        with open(f"{save_path}/article_{index}.html", 'w') as file:
            file.write(req.text)
    else:
        with open(f"{save_path}\article_{index}.html", 'w') as file:
            file.write(req.text)

def save_html_AnimePage_In_ListAnimePage(urls, folderNumber, CPUs = multiprocessing.cpu_count()):
    # This function saves the html of the anime of all the pages in folders, the anime present in a page are saved in a folder.
    # So I will have a folder for each page.
    
    # I divide the process into several sub-processes to not overload the computer.
    pool = ThreadPool(CPUs)
    # For each page I read I call the "save_html_animePage" function (previously defined) which saves the html of the anime in a folder.
    pool.map(lambda url : save_html_animePage(url, folderNumber, (50*(folderNumber-1)) + urls.index(url) +1), urls)


def get_listAnimePage(index, listPages):
    # With this function I download all the pages from "MyAnimeList".
    # Each page will contain 50 anime, that is 50 html which I will then have to save.
    
    listPages[index] = requests.get(f"https://myanimelist.net/topanime.php{'?limit={}'.format(50*index)}")
    
    if(listPages[index].status_code != 200) : 
        raise Exception(f"Web site have closed the connection at page: {index}")



def get_urls_In_ListAnimePage(page, pages):
    # In every page that I have downloaded I look for all the links of the souls present in a single page.
    # I take only those with the tag "href" which are exactly the links of the anime.
    
    soup = BeautifulSoup(page.content, "html.parser")
    
    # Find all links of the animes
    Urls = soup.find_all("a", class_="hoverinfo_trigger fl-l ml12 mr8", id=lambda string: string and string.startswith('#area'), href=True)
    
    #get just the href
    animeLinks = []
    for link in Urls:
        link_anime = str(link.get("href"))
        animeLinks.append(link_anime)
    
    pages[pages.index(page)] = animeLinks
    
    
def initGet(pageToGet = 383 ,CPUs = multiprocessing.cpu_count()):
    # This is the main function that starts the whole process. It takes as input the number of pages to download.
    
    pages = [None] * pageToGet
    numberOfPage = range(0, pageToGet)
    
    # I divide the process into several sub-processes to not overload the computer.
    pool = ThreadPool(CPUs)
    
    # For each page he reads, he creates the list of the 50 anime present in the page and for each page he saves the urls of the souls.
    # I do this using the two functions I defined previously.
    pool.map(lambda num : get_listAnimePage(num, pages), numberOfPage)   
    pool.map(lambda page : get_urls_In_ListAnimePage(page, pages), pages)
    
    with open("./generic_datafile/urls_anime.txt", "w") as file:
        for page in tqdm(pages):
            for url in page:
                file.write(str(url))
                file.write("\n")
    
    return pages


def getAnime(pages, start=0):
    # For each page I downloaded, I save the html of the 50 anime.
    # I do this using the previously defined "save_html_AnimePage_In_ListAnimePage" function.
    
    pages_from_start_to_end = pages[start:]
    for i in tqdm(range(0, len(pages_from_start_to_end))) : 
        save_html_AnimePage_In_ListAnimePage(pages_from_start_to_end[i], start+i+1)


"""
============================================================================================================

    parsing functions

============================================================================================================

"""

def findUsers(string):
    # With this function I find the "users".
    # The only place I can find "users" is in the "Score" block, where "users" is the third part of this block. So I just consider that part.
    #I notice that the number is written twice, once with commas and once without commas, so I have to fix it to get the correct data.
    
    string = string.split()[3].split(",")
    
    if(len(string) < 2):
        string[0] = string[0][:len(string[0])//2]
        
    if(len(string) == 2):
        temp = string[0].replace(string[1], "")
        temp = temp[:len(temp)//2]
        string.insert(0,temp)
        string.pop(1)
       
    
    if(len(string) == 3):
        temp = string[0].replace(string[1]+string[2], "")
        temp = temp[:len(temp)//2]
        string.insert(0,temp)
        string.pop(1)
        
    users_string = "".join(string)
    users_integer = int(users_string)
        
    return users_integer


def str_to_datetime(d): #Convert a string into a datetime type
    """Input: string,
    Output: list"""

    # With this function we can convert a string into a datetime type, but we need to check all possible combinations.

    if d=="Not available":
        return None
    else: 
        d = re.sub(r',', '', d) #first of all, remove the comma from the string
        d = re.sub(r' ','',d) #remove also the space
       
        if "to" in d:       
            date_time_list = d.split("to") #split the date of start and the date of end
            [start,end] = date_time_list[:]
            
            if len(start)==4: #if is only year
                start_datetime = datetime.strptime(start, "%Y").date()
            elif len(start)==7: #if is year and month
                start_datetime = datetime.strptime(start, "%b%Y").date()
            else:
                start_datetime = datetime.strptime(start, "%b%d%Y").date()
                
            
            if "?" in end:
                end_datetime = None
                return [start_datetime,end_datetime]
            else:
                
                if len(end)==4: #if is only year
                    end_datetime = datetime.strptime(end, "%Y").date()
                elif len(end)==7: #if is year and month
                    end_datetime = datetime.strptime(end, "%b%Y").date()
                else:
                    end_datetime = datetime.strptime(end, "%b%d%Y").date()
        
                return [start_datetime,end_datetime]
        
        else: #there is only the date of starting
            if len(d)==4: #if is only year
                start_datetime = datetime.strptime(d, "%Y").date()
            elif len(d)==7: #if is year and month
                start_datetime = datetime.strptime(d, "%b%Y").date()
            else:
                start_datetime = datetime.strptime(d, "%b%d%Y").date()
            return[start_datetime,start_datetime]



def tdForCharacters_Voices(tag):
    return tag.name == "td" and not tag.has_attr("width") and tag.has_attr("valign")


def getDataFromPage(pagePath):
    # This function takes an html as input and saves all the useful information present in the page.
    
    # I open the file that I pass as input for reading
    with open(pagePath, "r") as file:
        soup = BeautifulSoup(file, "html.parser")
        
    # I take all the "div" tag with class "spaceit_add" because this class contains all the information that interests me.
    temp = soup.find_all("div", {"class": "spaceit_pad"}) 

    out=[]   # I create a temporary list
    tempDict = collections.OrderedDict()   # I create a temporary dictionary
    finalDict = collections.OrderedDict()   # I create the final dictionary in which I will save all the information

    # With this for loop I scroll through all the "div" tags that I have taken
    # and I save in the "out" list all the texts present in these tags
    for i in temp:
        out.append(i.text)

    # In this "for" loop I go through all the elements in the out list and clean up those elements a bit.
    for i in range(0, len(out)):
        out[i] = out[i].strip()   # I eliminate the spaces present at the extremes
        out[i] = out[i].strip("\n")   # I delete the "\ n" present at the extremes
        out[i] = out[i].replace("\n", " ")   # I replace the "\ n" present within the text with a space


    # Now I create the temporary dictionary with the information I have collected.
    # Then I start a "for" loop on all the elements of the "out" list.
    # For each element of the list I look for the index of ":".
    # In this way I can create the key of my dictionary and its value.
    for i in out:
        index = i.find(":")
        tempDict[i[:index]] = i[index+1:].strip()


    # Now we have to clean up the data we collected and then create the final dictionary.
    # For each element of the temporary dictionary I have to check if it exists or not, 
    # or if it has particular values, for example "N / A", "Not available", ... 
    # If it exists I add it to the final dictionary, 
    # if it does not exist I add "None "to indicate that the value is undefined.
    if(tempDict["Type"] != "N/A"):
        finalDict["Type"] = tempDict["Type"]#Anime Type
    else:
        finalDict["Type"] = ""

    if(tempDict["Episodes"] != "N/A" and tempDict['Episodes'] != 'Unknown'):
        finalDict["Episodes"] = int(tempDict["Episodes"]) #number of episodes
    else:
        finalDict["Episodes"] = None
        
    # In this case we have to use the "str_to_datetime" function we defined previously 
    # to transform all the strings indicating dates into a datetime.
    # We separate the starting date from the ending date.
    if(tempDict["Aired"] != "N/A" and tempDict["Aired"] != "Not available"):
        aired = str_to_datetime(tempDict["Aired"])
        finalDict["releasedDate"]  = aired[0]
        finalDict["endDate"] = aired[1]
    else:
        finalDict["Aired"] = None

    if(tempDict["Members"].replace(",", "") != "N/A"):
        finalDict["Members"] = int(tempDict["Members"].replace(",", "")) #members

    else:
        finalDict["Members"] = None

    # I find the users in the "Score" block.
    # I do this using the "findUsers" function that I defined earlier.
    if(tempDict["Score"].split()[3] != "-"):
        finalDict["Users"] = findUsers(tempDict["Score"])# Users
    else:
        finalDict["Users"] = None

    if(tempDict["Score"].split()[0][:-1] != "N/A"):
        finalDict["Score"] =  float(tempDict["Score"].split()[0])#Score
    else:
        finalDict["Score"] = None

    if(tempDict["Ranked"].split()[0].strip("#")[:-1] != "N/A"):
        finalDict["Ranked"] = int(tempDict["Ranked"].split()[0].strip("#")[:-1])  #Rank
    else:
        finalDict["Ranked"] = None

    if(tempDict["Popularity"] != "N/A"):
        finalDict["Popularity"] = int(tempDict["Popularity"].strip("#")) #Popularity
    else:
        finalDict["Popularity"] = None


    #Characters
    #Voices
    #Staff

    #ADDING THE NAME
    temp = soup.find("strong") 

    finalDict["Name"] = temp.text

    #ADDING THE SYNOPSIS
    temp = soup.find("p", itemprop="description")

    if(temp.text != "No synopsis information has been added to this title. Help improve our database by adding a synopsis here."):
        finalDict["Synopsis"] = temp.text.replace("\n", " ")
    else:
        finalDict["Synopsis"] = None

    #ADDING THE RELATED ANIME
    # This variable is not always defined so I have to do a test.
    # If it exists, I take the link, I use "set" because I don't want repetitions and then add it to the dictionary.
    # If it doesn't exist I add "None" to the dictionary.
    try:
        temp = soup.find("table", class_="anime_detail_related_anime", style="border-spacing:0px;")
        temp = temp.find_all("a")

        yt = set()

        for t in temp:
            yt.add(t.text)

        finalDict["Related_Anime"] = list(yt)

    except:
        finalDict["Related_Anime"] = None

    #FIND CHARACTERS, VOICES, STAFF AND ROLE
    characters = []
    voices = []
    staff = []
    role = []

    # I can save all these elements together because they are all in the same table.
    # I consider first the left column, then the right column.
    # I check if they exist or not and then add them to the dictionary.
    try:
        temp = soup.find_all("div", class_="left-column fl-l divider")

        temp0 = temp[0].find_all("table", width="100%")

        for t in temp0:
            t = t.find_all(tdForCharacters_Voices)

            try:
                characters.append(t[0].find("a").string)
            except:
                characters.append("")
            try:
                voices.append(t[1].find("a").string)
            except:
                voices.append("")

        temp1 = temp[1].find_all("table", width="100%")


        for t in temp1:
            t = t.find(tdForCharacters_Voices)

            try:
                staff.append(t.find("a").string)
            except:
                staff.append("")

            try:
                role.append(t.find("small").string)
            except:
                role.append("")

    except:
        pass

    try:
        temp = soup.find_all("div", class_="left-right fl-r")

        temp0 = temp[0].find_all("table", width="100%")

        for t in temp0:
            t = t.find_all(tdForCharacters_Voices)

            try:
                characters.append(t[0].find("a").string)
            except:
                characters.append("")
            try:
                voices.append(t[1].find("a").string)
            except:
                voices.append("")


        temp1 = temp[1].find_all("table", width="100%")


        for t in temp1:
            t = t.find(tdForCharacters_Voices)

            try:
                staff.append(t.find("a").string)
            except:
                staff.append("")

            try:
                role.append(t.find("small").string)
            except:
                role.append("")

    except:
        characters = None
        voices = None
        staff = None
        role = None

    finalDict["Characters"] = characters
    finalDict["Voices"] = voices
    finalDict["Staffs"] = staff
    finalDict["Roles"] = role

    # In the end I get the final dictionary for a single anime.
    
    return finalDict


"""
============================================================================================================

    Path functions

============================================================================================================

"""
# These are two functions that I need to create a list of strings indicating the path of each file.

def animeFile_path():
    # This function saves the path where each anime is saved.
    
    animePath = []

    for animeDir in range(1,384):
        for animePage in range(1,51):
            try:
                with open(f'./animeList_pages/{animeDir}th_page/article_{animePage + ((animeDir-1)*50)}.html', 'r') as file:
                    pass
                animePath.append(f'./animeList_pages/{animeDir}th_page/article_{animePage + ((animeDir-1)*50)}.html')
            except:
                pass
    return animePath


def anime_tsv_path(control_parameter = 10):
    # This function saves the path where each tsv file related to anime is saved.
    
    anime_tsv_path = []
    check = 0
    index = 1

    while(check < control_parameter):
        try:
            with open(f'./anime_tsv/anime_{index}.tsv', 'r') as file:
                pass
            anime_tsv_path.append(f'./anime_tsv/anime_{index}.tsv')
        except:
            check += 1
        
        index += 1
        
    return anime_tsv_path


"""
============================================================================================================

    TSV functions

============================================================================================================

"""


def write_anime_tsv(pagePath):
    
    # I call the "getDataFromPage" function created previously which collects all the data of an anime in a single dictionary.
    data = getDataFromPage(pagePath)
    
    #Use regex to find the anime number
    pattern = re.compile("_[0-9].*")
    index = pattern.findall(pagePath)[0].strip('_html.')
    
    # I create the path in whiche I want to save these tsv files and I save them.
    save_path = f"{pathlib.Path().resolve()}/anime_tsv"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    if(sys.platform != "win32"):
        with open(f"{save_path}/anime_{index}.tsv", "w") as file:
            tsv_writer = csv.writer(file, delimiter='\t')

            tsv_writer.writerow(data.keys())

            tsv_writer.writerow(data.values())
    else:
        with open(f"{save_path}\anime_{index}.tsv", "w") as file:
            tsv_writer = csv.writer(file, delimiter='\t')

            tsv_writer.writerow(data.keys())

            tsv_writer.writerow(data.values())



def write_all_anime_tsv(CPUs = multiprocessing.cpu_count()):
    # Now I write all the tsv files concerning the anime.

    # I divide the process into several sub-processes to not overload the computer.
    pool = ThreadPool(CPUs)

    # I get all the folders where the anime are saved using the "animeFile_path" function that I created earlier.
    anime = animeFile_path()

    # For each anime in the anime list I save the tsv file.
    pool.map(lambda anime: write_anime_tsv(anime), anime);
    
    
"""
============================================================================================================

    preprocessing functions

============================================================================================================

"""

def preprocessing(string):
    processed_string = []
    
    # I initialize the stemmer.
    stemmer = nltk.SnowballStemmer("english", ignore_stopwords=False)
    
    # First I tokenize the string I take as input, that is, I create a list of words and symbols.
    words = nltk.word_tokenize(string)
    
    words = list(map(lambda word: word.replace("°", ""), words))
    
    # Then I use a "for" loop to eliminate all symbols and punctuation to get a list of only words and numbers.
    new_words= [word for word in words if word.isalnum()]
    
     # Now I stemming on the new list of words I created.
    for word in new_words:
        processed_string.append(stemmer.stem(word))
     
    # I don't want repetitions, so I use "set"
    processed_string = set(processed_string)
    
     # At the end I get all the words that will go into the dictionary.
     
    return processed_string

def preprocessing_with_occurences(string):
    # This function does the same thing as the previous function, 
    # but in this case I don't use "set" because I want to consider repetitions.
    
    processed_string = []
    stemmer = nltk.SnowballStemmer("english", ignore_stopwords=False)
    words = nltk.word_tokenize(string)
    new_words= [word for word in words if word.isalnum()]
    
    
    for word in new_words:
        processed_string.append(stemmer.stem(word))
    
    return processed_string

"""
============================================================================================================

    indexing functions

============================================================================================================

"""

def make_inverted_index():
    # I access my dataset in the "Synopsis" column
    data = pd.read_csv("./generic_datafile/dataset.csv", usecols=["Synopsis"])
    
    # I create a dictionary in which I will save my "inverted index".
    dic = collections.defaultdict(list)
    
    # index = document number ; synops = corpus of the document
    for index, synops in tqdm(enumerate(data["Synopsis"].array)):
        # First of all I check that the synopsis is a string (because not all anime have the synopsis)
        if(isinstance(synops, str)):
            # For each word present in the corpus of my document I do the "preprocessing" (function created previously), that is, I create a list of words.
            # Then I transform each word in the list into a hash, add it to my dictionary and append the number of the document in which it appears.
            # In this way every time I add the number of the document if a word is already present in the dictionary and otherwise it creates a new one.
            for word in preprocessing(synops):
                dic[hashlib.sha256(word.encode()).hexdigest()].append(str(index+1))
    
    # I save my dictionary in a file and this will be my "inverted index".
    with open("./generic_datafile/inverted_index.txt", "w") as file:
        for key in tqdm(dic):
            file.write(str(key) + ":" + ",".join(dic[key]))
            file.write("\n")
            


def make_inverted_index_tfidf():
    # This function is similar to the prevoius function "make_inverted_index", 
    # but in this case I use the function "tfidf" to calculate the inverted index.
    
    vocabulary = read_vocabulary_per_doc()
    synopsis = read_synopsis()
    inverted_index_new = collections.defaultdict(list)
    inverted_index_old = read_inverted_index()
    
    for word in tqdm(vocabulary):
        inverted_index_new[word[0]].append(str(tfidf(word[0], int(word[1])-1, synopsis, inverted_index_old)))

    
    with open("./generic_datafile/inverted_index_tfidf.txt", "w") as file:
        for key in tqdm(inverted_index_new):
            file.write(key + ":" + ";".join(inverted_index_new[key]))
            file.write("\n")

            
def make_inverted_index_tfidf_with_names():
    # This function is equal to the previous function "make_inverted_index_tfidf", 
    # the only difference is that in this case I consider also the name of the anime.
    
    vocabulary = read_vocabulary_per_doc_with_names()
    synopsis = read_synopsis_and_names()
    inverted_index_new = collections.defaultdict(list)
    inverted_index_old = read_inverted_index_with_names()
    
    for word in tqdm(vocabulary):
        inverted_index_new[word[0]].append(str(tfidf(word[0], int(word[1])-1, synopsis, inverted_index_old)))

    
    with open("./generic_datafile/inverted_index_tfidf_with_name.txt", "w") as file:
        for key in tqdm(inverted_index_new):
            file.write(key + ":" + ";".join(inverted_index_new[key]))
            file.write("\n")
            
            
def make_inverted_index_with_names():
    # This function is equal to the previous function "make_inverted_index", but in this case I consider also the name of the anime.
    
    data = pd.read_csv("./generic_datafile/dataset.csv", usecols=["Synopsis", "Name"])
    dic = collections.defaultdict(list)
    
    for index, synops in tqdm(enumerate(data["Synopsis"].array)):
        if(isinstance(synops, str)):
            for word in preprocessing(synops):
                dic[hashlib.sha256(word.encode()).hexdigest()].append(str(index+1))
                
    for index, name in tqdm(enumerate(data["Name"].array)):
        if(isinstance(name, str)):
            for word in preprocessing(name):
                dic[hashlib.sha256(word.encode()).hexdigest()].append(str(index+1))
    
    with open("./generic_datafile/inverted_index_with_name.txt", "w") as file:
        for key in tqdm(dic):
            file.write(str(key) + ":" + ",".join(dic[key]))
            file.write("\n")
            

"""
============================================================================================================

    score functions

============================================================================================================

"""

def tfidf(i,j,documents, dic):
    """
    Input: (string, int, pandas.array)
    Output: int
    """
    
    n_ij=documents[j].count(i)
    
    
    tfij=n_ij/len(documents[j])
    
    idf_den = len(dic[i])
  
    idf=np.log10(len(documents)/idf_den)
    
    return (j+1,tfij*idf)



def tfidf_query(word, doc, documents, inverted_index_old):
    
    n_ij=doc.count(word)
    
    tfij=n_ij/len(doc)
   
    idf_den = len(inverted_index_old[word])
    
    idf=np.log10(len(documents)/idf_den)
   
    return tfij * idf




def cosine_similarity(query, doc, vocabulary, documents, inverted_index_old):
    
    N = np.zeros(len(vocabulary))
    
    for i, word in enumerate(vocabulary):
        try:
            N[i] = tfidf_query(word, query, documents, inverted_index_old)
        except KeyError:
            N[i] = 0.0
    
    try:
        doc = read_doc(doc, len(vocabulary))
        
    
        similarity = (np.dot(doc,N)/(np.linalg.norm(doc) * np.linalg.norm(N)))
    
    except:
        similarity = 0.0
    
    return similarity


def calculate_score(query, doc_index, vocabulary, synopsis, inverted_index_old):
    # I define a new score to do my research.
    
    names, types, popularity = read_processed_columns()
    cos_sim = cosine_similarity([sha256(w) for w in query], doc_index, vocabulary, synopsis, inverted_index_old)
    
    name_score = len(names[doc_index].intersection(set(query)))/len(names[doc_index]) # Value between 0 and 1
    
    popularity_score = ((max(popularity.values())-popularity[doc_index])/max(popularity.values()))*2 # Value between 0 and 2  (independent of the query)
    
    # If a particular type is entered (for example TV, episodes, ...), 
    # I also calculate how many elements have that type, in order to have a better result.
    if(types[doc_index] in query):
        types_score = 1
    else:
        types_score = 0
        
    return (cos_sim + name_score + popularity_score + types_score)



"""
============================================================================================================

    Search engine functions

============================================================================================================

"""


def search():
    # I open the dataset on which I want to do my research.
    tsv_data = pd.read_csv("./generic_datafile/dataset.csv")[["Name", "Synopsis", "url"]]
    # I load my "inverted index" (function created below).
    dic = read_inverted_index()
    # I define a variable to terminate the loop.
    stop = False
    # I initialize a dataframe which will be my output.
    out_data = pd.DataFrame()
    
    while(not stop):
        print("INSTRUCTIONS:\nto close the search engine:exit()\n\n")
        
        # I ask the user to enter the query.
        query = input("Search:")

        if(query == "exit()"):
            stop = True
            print("\n\n\nstopping search engine...\n\n\n")
        else:
            # First I use the "preprocessing" function to transform the query into a list of words.
            query = list(preprocessing(query))
            
            # I must distinguish two cases because if the query that is entered is not present in my dictionary I will not have a result.
            try:
                indexs = []
                # I first convert the query words to hash. and I create a new list.
                for w in query:
                    indexs.append(set(dic[sha256(w)]))

                # I make the intersection because I want to see in which document all the words of the query are contained.
                indexs = indexs[0].intersection(*indexs)
                
                # For each word in the new list I check which document it is in and I build the final dataframe.
                if(len(indexs) != 0):
                    for index in indexs:
                        out_data = out_data.append(tsv_data.loc[int(index)-1], ignore_index=True)
                        
                    display(out_data)   # I show the final dataframe
                
                else:
                    print("\nNo results\n\n")

                # I free the dataframe to start a new search
                out_data.drop(out_data.index[:], inplace=True)
                
            except KeyError:
                print("\nNo results\n\n")



    
def search_2():
    # This function is equal to the previous function "search", 
    # the only difference is that in this case I use the cosine similarity to calculate the distance.
    
    tsv_data = pd.read_csv("./generic_datafile/dataset.csv")[["Name", "Synopsis", "url"]]
    dic = read_inverted_index_tfidf()
    inverted_index_old = read_inverted_index()
    stop = False
    out_data = pd.DataFrame()
    vocabulary = read_vocabulary()
    synopsis = read_synopsis()
    
    while(not stop):
        print("INSTRUCTIONS:\nto close the search engine:exit()\n\n")
        
        query = input("Search:")

        if(query == "exit()"):
            stop = True
            print("\n\n\nstopping search engine...\n\n\n")
        else:
            query = list(preprocessing(query))
            try:
                indexs = []
                freqs = []
                similarities = []
                
                for w in query:
                    indexs.append(set([i[0] for i in dic[sha256(w)]]))

                indexs = indexs[0].intersection(*indexs)

                for i in indexs:
                    similarities.append(cosine_similarity([sha256(w) for w in query], i, vocabulary, synopsis, inverted_index_old))
                
                similarities_and_docs = list(zip(similarities, indexs))
                
                similarities_and_docs = maxHeapfy(similarities_and_docs)
                

                for index in similarities_and_docs:
                    out_data = out_data.append(tsv_data.loc[int(index[1])-1], ignore_index=False)

                out_data["Similarity"] = [simil[0] for simil in similarities_and_docs]

                
                display(out_data)
                
                out_data.drop(out_data.index[:], inplace=True)
                
            except KeyError:
                print("\nNo results\n\n")
                


def search_3():
    # This function is equal to the previous function "search_2", 
    # but in this case I use the new score that I have defined in the function "calculate_score" 
    
    tsv_data = pd.read_csv("./generic_datafile/dataset.csv")[["Name", "Synopsis", "url"]]
    dic = read_inverted_index_tfidf_with_names()
    inverted_index_old = read_inverted_index_with_names()
    stop = False
    out_data = pd.DataFrame()
    vocabulary = read_vocabulary()
    synopsis = read_synopsis_and_names()
    
    while(not stop):
        print("INSTRUCTIONS:\nto close the search engine:exit()\n\n")
        
        query = input("Search:")

        if(query == "exit()"):
            stop = True
            print("\n\n\nstopping search engine...\n\n\n")
        else:
            query = list(preprocessing(query))
            try:
                indexs = []
                freqs = []
                similarities = []
                
                for w in query:
                    indexs.append(set([i[0] for i in dic[sha256(w)]]))

                indexs = indexs[0].intersection(*indexs)

                for i in indexs:
                    similarities.append(calculate_score(query, i, vocabulary, synopsis, inverted_index_old))
                
                similarities_and_docs = list(zip(similarities, indexs))
                
                similarities_and_docs = maxHeapfy(similarities_and_docs)

                for index in similarities_and_docs:
                    out_data = out_data.append(tsv_data.loc[int(index[1])-1], ignore_index=False)

                out_data["Similarity"] = [simil[0] for simil in similarities_and_docs]

                
                display(out_data)
                
                out_data.drop(out_data.index[:], inplace=True)
                
            except KeyError:
                print("\nNo results\n\n")
    
    
"""
============================================================================================================

    file handle functions

============================================================================================================

"""
# With these functions I read the various documents that I have saved in my folders.

def read_inverted_index(path = "./generic_datafile/inverted_index.txt"):
    
    dic = dict()
    
    with open(path, "r") as file:
        lines = file.read().splitlines()
        for line in lines:
            key = line.split(":")[0]
            values = line.split(":")[1].split(",")
            dic[key] = values

    return dic

def read_inverted_index_with_names(path = "./generic_datafile/inverted_index_with_name.txt"):
    
    dic = dict()
    
    with open(path, "r") as file:
        lines = file.read().splitlines()
        for line in lines:
            key = line.split(":")[0]
            values = line.split(":")[1].split(",")
            dic[key] = values

    return dic

def read_inverted_index_tfidf(path = "./generic_datafile/inverted_index_tfidf.txt"):
    
    dic = dict()
    
    with open(path, "r") as file:
        lines = file.read().splitlines()
        for line in lines:
            key = line.split(":")[0]
            values = line.split(":")[1].split(";")
            values = [el.strip("()") for el in values]
            values = [tuple(map(float, i.split(','))) for i in values]
            dic[key] = values

    return dic


def read_inverted_index_tfidf_with_names(path = "./generic_datafile/inverted_index_tfidf_with_name.txt"):
    
    dic = dict()
    
    with open(path, "r") as file:
        lines = file.read().splitlines()
        for line in lines:
            key = line.split(":")[0]
            values = line.split(":")[1].split(";")
            values = [el.strip("()") for el in values]
            values = [tuple(map(float, i.split(','))) for i in values]
            dic[key] = values

    return dic


def read_vocabulary_per_doc(path = "./generic_datafile/vocabulary_per_doc.txt"):
    
    voc = []

    with open(path, "r") as file:
        lines = file.read().splitlines()
        for line in lines:
            voc.append((line.split(":")[0], line.split(":")[1]))
        

    return voc

def read_vocabulary_per_doc_with_names(path = "./generic_datafile/vocabulary_per_doc_with_names.txt"):
    
    voc = []

    with open(path, "r") as file:
        lines = file.read().splitlines()
        for line in lines:
            voc.append((line.split(":")[0], line.split(":")[1]))
        

    return voc

def read_vocabulary(path = "./generic_datafile/vocabulary.txt"):
    
    voc = collections.OrderedDict()

    with open(path, "r") as file:
        lines = file.read().splitlines()
        for line in lines:
            voc[line.split(":")[0]] = 0.0
            
    return voc

def read_synopsis():
    with open('./generic_datafile/synopsis.plk', 'rb') as file:
        out = pickle.load(file)
    return out

def read_synopsis_and_names():
    with open('./generic_datafile/synopsis_and_names.plk', 'rb') as file:
        out = pickle.load(file)
    return out


def read_doc(i, length):
    
    arr = np.zeros(length)
    
    with open(f"./generic_datafile/bags_of_words/doc_{int(i)}.txt") as file:
        arr = np.array([ float(line.split(":")[1]) for line in file.read().splitlines()])
    
    return arr


def read_processed_columns():
    with open('./generic_datafile/names.plk', 'rb') as file:
        names = pickle.load(file)
        
        
    with open('./generic_datafile/types.plk', 'rb') as file:
        types = pickle.load(file)
        
        
    with open('./generic_datafile/popularity.plk', 'rb') as file:
        popularity = pickle.load(file)
        
    return names, types, popularity



"""
============================================================================================================

    vocabulary functions

============================================================================================================

"""



#Parole si ripetono ma solo in documenti diversi
def make_vocabulary_of_every_word_in_doc():
    data = pd.read_csv("./generic_datafile/dataset.csv", usecols=["Synopsis", "Name"])
    list_of_processed_synops = []
    
    for index, synops in tqdm(enumerate(data["Synopsis"].array)):
        if(isinstance(synops,str)):
            list_of_processed_synops += [[word ,index+1] for word in preprocessing(synops)]
    
    for index, word in tqdm(enumerate(list_of_processed_synops)):
        list_of_processed_synops[index][0] = hashlib.sha256(word[0].encode()).hexdigest()
        
    
    with open("./generic_datafile/vocabulary_per_doc.txt", "w") as file:
        for word in list_of_processed_synops:
            file.write(word[0] + ":" + str(word[1]))
            file.write("\n")


def make_vocabulary_of_every_word_in_doc_with_names():
    data = pd.read_csv("./generic_datafile/dataset.csv", usecols=["Synopsis", "Name"])
    list_of_processed_synops_and_name = []
    
    for index, synops in tqdm(enumerate(data["Synopsis"].array)):
        if(isinstance(synops,str)):
            list_of_processed_synops_and_name += [[word ,index+1] for word in preprocessing(synops)]
        
    print(len(list_of_processed_synops_and_name))
            
    for index, name in tqdm(enumerate(data["Name"].array)):
        if(isinstance(name,str)):
            list_of_processed_synops_and_name += [[word ,index+1] for word in preprocessing(name)]
            
    print(len(list_of_processed_synops_and_name))
    
    for index, word in tqdm(enumerate(list_of_processed_synops_and_name)):
        list_of_processed_synops_and_name[index][0] = hashlib.sha256(word[0].encode()).hexdigest()
        
    
    with open("./generic_datafile/vocabulary_per_doc_with_names.txt", "w") as file:
        for word in list_of_processed_synops_and_name:
            file.write(word[0] + ":" + str(word[1]))
            file.write("\n")


def make_vocabulary():
    voc_per_doc = read_vocabulary_per_doc()
    
    vocabulary = set()
    
    for tupla in voc_per_doc:
        vocabulary.add(tupla[0])
    
    with open("./generic_datafile/vocabulary.txt", "w") as file:
        for word in vocabulary:
            file.write(word)
            file.write("\n")
            




"""
============================================================================================================

    bags of words functions

============================================================================================================

"""
            
    
def make_bagOfWords():
    data = pd.read_csv("./generic_datafile/dataset.csv", usecols=["Synopsis"])
    synopsis = read_synopsis()
    inverted_index_old = read_inverted_index()
    vocabulary = read_vocabulary()
    
    for i, synops in tqdm(enumerate(data["Synopsis"].array)):
        if(isinstance(synops,str) and synops != " "):
            make_single_bagOfWord(vocabulary, i, synopsis, inverted_index_old)
            
            
            
def make_single_bagOfWord(vocabulary, doc_index, documents, inverted_index):
    
    bag = collections.OrderedDict()
    
    for word in vocabulary:
        bag[word] = tfidf(word, doc_index, documents, inverted_index)[1]
        
    with open(f"./generic_datafile/bags_of_words/doc_{doc_index+1}.txt", "w") as file:
        for word in bag:
            file.write(word + ":" + str(bag[word]))
            file.write("\n")

            
"""
============================================================================================================

    dataset functions

============================================================================================================

"""


def make_dataframe(animePath = anime_tsv_path()):
    # This function takes as input the path of the tsv file of the anime and creates a single pandas tsv file 
    # with all the data I have collected on the anime.
    
    # I create the tsv file by adding the first anime tsv file.    
    tsv_data = pd.read_csv(animePath[0], sep = "\t") 
    
    # Then I start a for loop with which I add all the other tsv files, one below the other.
    for path in tqdm(animePath[1:]):
        tsv_data = tsv_data.append(pd.read_csv(path, sep ="\t"), ignore_index=True)
    
    # I also add the column with the url of the pages.
    with open("./generic_datafile/urls_anime.txt", "r") as file:
        lines = file.read().splitlines()
        tsv_data['url'] = np.resize(lines,len(tsv_data))

    # In the end I will have a single tsv file containing all the information.
    tsv_data.to_csv("./generic_datafile/dataset.csv")
        
    return tsv_data


def preprocess_dataset_columns():
    # In this function I use the funtion "preprocessing" that I have defined previously 
    # to create a list of words using the columns "Name", "Popularity" and "Type.
    
    data = pd.read_csv("./generic_datafile/dataset.csv", usecols=["Name", "Popularity", "Type"])
    
    names_data = data["Name"].array
    popularity_data = np.array(data["Popularity"].array)
    type_data = data["Type"].array
    
    
    names = collections.OrderedDict()
    
    for doc_index, w in tqdm(enumerate(names_data), desc = "names"):
        names[doc_index+1] = f.preprocessing(w)
        
    with open("./generic_datafile/names.plk", 'wb') as file:  # Overwrites any existing file.
        pickle.dump(names, file, pickle.HIGHEST_PROTOCOL)
        
    types = collections.OrderedDict()
    
    for doc_index, t in tqdm(enumerate(type_data), desc = "types"):
        types[doc_index+1] = f.preprocessing(t)
        
    with open("./generic_datafile/types.plk", 'wb') as file:  # Overwrites any existing file.
        pickle.dump(types, file, pickle.HIGHEST_PROTOCOL)
        
    popularity = collections.OrderedDict()
    
    for doc_index, p in tqdm(enumerate(popularity_data), desc = "popularity"):
        popularity[doc_index+1] = p
        
    with open("./generic_datafile/popularity.plk", 'wb') as file:  # Overwrites any existing file.
        pickle.dump(popularity, file, pickle.HIGHEST_PROTOCOL)
        

"""
============================================================================================================

    list of words per document functions

============================================================================================================

"""


def processed_synopsis():
    # I use the preprocessing function to transform the synopsis into a list of words which I will then use to do my research.
    
    data = pd.read_csv("./generic_datafile/dataset.csv", usecols=["Synopsis"])
    list_of_processed_synops = []
    
    for synops in tqdm(data["Synopsis"].array):
        if(isinstance(synops,str)):
            list_of_processed_synops.append(preprocessing_with_occurences(synops))
        else:
            list_of_processed_synops.append([])
    
    for index, synops in tqdm(enumerate(list_of_processed_synops)):
        hashwords = []
        for word in synops:
            hashwords.append(hashlib.sha256(word.encode()).hexdigest())
        list_of_processed_synops[index] = hashwords
    
    with open("./generic_datafile/synopsis.plk", 'wb') as file:  # Overwrites any existing file.
        pickle.dump(list_of_processed_synops, file, pickle.HIGHEST_PROTOCOL)
        
        
def processed_synopsis_and_names():
    # This function is equal to the previous function, but in this case I precess also the name of the anime.
    data = pd.read_csv("./generic_datafile/dataset.csv", usecols=["Synopsis", "Name"])
    list_of_processed_synops_and_names = []
    
    for synops, name in tqdm(zip(data["Synopsis"].array, data["Name"].array)):
        if(isinstance(synops,str) or isinstance(name,str)):
            stringToProcess = str(synops) + " " + name
            list_of_processed_synops_and_names.append(preprocessing_with_occurences(stringToProcess))
        else:
            list_of_processed_synops_and_names.append([])
            
    
    for index, synops in tqdm(enumerate(list_of_processed_synops_and_names)):
        hashwords = []
        for word in synops:
            hashwords.append(hashlib.sha256(word.encode()).hexdigest())
        list_of_processed_synops_and_names[index] = hashwords
    
    with open("./generic_datafile/synopsis_and_names.plk", 'wb') as file:  # Overwrites any existing file.
        pickle.dump(list_of_processed_synops_and_names, file, pickle.HIGHEST_PROTOCOL)
        
        

"""
============================================================================================================

    miscellaneous functions

============================================================================================================

"""

def maxHeapfy(lst):
    
    negative_lst = []
    
    # I change the sign to all the elements of the list because the "heappop" function that I will use later finds the minimum, 
    # so if I change the sign to all the elements and calculate the minimum, I actually find the maximum of the original list.
    for elem in lst:
        negative_lst.append((-elem[0], elem[1]))
    
    heapq.heapify(negative_lst)
    
    negative_lst = [heapq.heappop(negative_lst) for i in range(len(negative_lst))]
    
    sorted_lst = []
    
    for elem in negative_lst:
        sorted_lst.append((abs(elem[0]), elem[1]))
    
    return sorted_lst
        
def sha256(string):
    # This functions transforms the string into a hash.
    return str(hashlib.sha256(string.encode()).hexdigest())


def initialize_file_for_search_engine():
    
    
    print("Starting creating the tsv files...")
    write_all_anime_tsv()
    print("tsv files created")
    
    print("Starting creating the database...")
    make_dataframe()
    print("database created")
    
    
    print("Starting creating the inverted index...")
    make_inverted_index()
    print("Inverted index created")
    
    
    print("Starting creating the inverted index with names...")
    make_inverted_index_with_names()
    print("Inverted index created")
    
    
    print("Starting creating the vocabolary of every word per document...")
    make_vocabulary_of_every_word_in_doc()
    print("Vocabolary created")
    
    
    print("Starting creating the vocabolary of every word per document with names...")
    make_vocabulary_of_every_word_in_doc_with_names()
    print("Vocabolary created")
    
    
    print("Starting created the list of tokenized synopsis...")
    processed_synopsis()
    print("Tokenized synipsis created")
    
    print("Starting created the list of tokenized synopsis and names...")
    processed_synopsis_and_names()
    print("Tokenized synipsis created")
    
    
    print("Starting creating the vocabulary...")
    make_vocabulary()
    print("Vocabulary created")
    
    
    print("Starting creating the inverted index with TFIDF score...")
    make_inverted_index_tfidf()
    print("Inverted index created")
    
    
    print("Starting creating the inverted index with TFIDF score (with names)...")
    make_inverted_index_tfidf_with_names()
    print("Inverted index created")
    
    
    print("Starting creating the bags of words...")
    make_bagOfWords()
    print("Bags of words created")
    
    
    print("Startng processing the dataset columns...")
    preprocess_dataset_columns()
    print("Columns processed")
    
    
    print("All operation done.")
    
    

        
        
            
            
            
        
        
        
        
        
        