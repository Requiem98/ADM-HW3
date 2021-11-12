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

"""
============================================================================================================

    html pages download functions

============================================================================================================

"""

def save_html_animePage(url, directoryNum, index):
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
    pool = ThreadPool(CPUs)
    pool.map(lambda url : save_html_animePage(url, folderNumber, (50*(folderNumber-1)) + urls.index(url) +1), urls)


def get_listAnimePage(index, listPages):
    listPages[index] = requests.get(f"https://myanimelist.net/topanime.php{'?limit={}'.format(50*index)}")
    
    if(listPages[index].status_code != 200) : 
        raise Exception(f"Web site have closed the connection at page: {index}")



def get_urls_In_ListAnimePage(page, pages):
    
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

    pages = [None] * pageToGet
    numberOfPage = range(0, pageToGet)

    pool = ThreadPool(CPUs)
    pool.map(lambda num : get_listAnimePage(num, pages), numberOfPage)   
    pool.map(lambda page : get_urls_In_ListAnimePage(page, pages), pages)
    
    with open("./generic_datafile/urls_anime.txt", "w") as file:
        for page in tqdm(pages):
            for url in page:
                file.write(str(url))
                file.write("\n")
    
    return pages


def getAnime(pages, start=0):
    pages_from_start_to_end = pages[start:]
    for i in tqdm(range(0, len(pages_from_start_to_end))) : 
        save_html_AnimePage_In_ListAnimePage(pages_from_start_to_end[i], start+i+1)


"""
============================================================================================================

    parsing functions

============================================================================================================

"""

def findUsers(string):
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
    with open(pagePath, "r") as file:
        soup = BeautifulSoup(file, "html.parser")

    temp = soup.find_all("div", {"class": "spaceit_pad"}) 

    out=[]
    tempDict = collections.OrderedDict()
    finalDict = collections.OrderedDict()

    for i in temp:
        out.append(i.text)

    for i in range(0, len(out)):
        out[i] = out[i].strip()
        out[i] = out[i].strip("\n")
        out[i] = out[i].replace("\n", " ")



    for i in out:
        index = i.find(":")
        tempDict[i[:index]] = i[index+1:].strip()



    if(tempDict["Type"] != "N/A"):
        finalDict["Type"] = tempDict["Type"]#Anime Type
    else:
        finalDict["Type"] = ""

    if(tempDict["Episodes"] != "N/A" and tempDict['Episodes'] != 'Unknown'):
        finalDict["Episodes"] = int(tempDict["Episodes"]) #number of episodes
    else:
        finalDict["Episodes"] = None

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

    return finalDict


"""
============================================================================================================

    Path functions

============================================================================================================

"""


def animeFile_path():
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
    data = getDataFromPage(pagePath)
    
    #Use regex to find the anime number
    pattern = re.compile("_[0-9].*")
    index = pattern.findall(pagePath)[0].strip('_html.')
    
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

    pool = ThreadPool(CPUs)

    anime = animeFile_path()

    pool.map(lambda anime: write_anime_tsv(anime), anime);
    
    
"""
============================================================================================================

    preprocessing and indexing functions

============================================================================================================

"""

def preprocessing(string):
    processed_string = []
    stemmer = nltk.SnowballStemmer("english", ignore_stopwords=False)
    words = nltk.word_tokenize(string)
    new_words= [word for word in words if word.isalnum()]
    
    
    for word in new_words:
        processed_string.append(stemmer.stem(word))
        
    processed_string = set(processed_string)
    
    return processed_string


def make_inverted_index():
    data = pd.read_csv("./generic_datafile/dataset.csv", usecols=["Synopsis"])
    dic = collections.defaultdict(list)
    
    for index, synops in tqdm(enumerate(data["Synopsis"].array)):
        if(isinstance(synops, str)):
            for word in preprocessing(synops):
                dic[hashlib.sha256(word.encode()).hexdigest()].append(str(index+1))
    
    with open("./generic_datafile/inverted_index.txt", "w") as file:
        for key in tqdm(dic):
            file.write(str(key) + ":" + ",".join(dic[key]))
            file.write("\n")
            

"""
============================================================================================================

    Search engine functions

============================================================================================================

"""


def search():
    tsv_data = pd.read_csv("./generic_datafile/dataset.csv")[["Name", "Synopsis", "url"]]
    dic = read_dict()
    stop = False
    out_data = pd.DataFrame()
    
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
                for w in query:
                    indexs.append(dic[str(hashlib.sha256(w.encode()).hexdigest())])

                for index in indexs:
                    for j in index:
                        out_data = out_data.append(tsv_data.loc[int(j)-1], ignore_index=True)


                display(out_data)
                out_data.drop(out_data.index[:], inplace=True)
                
            except KeyError:
                print("\nNo results\n\n")



    
"""
============================================================================================================

    Miscellaneous functions

============================================================================================================

"""



def make_dataframe(animePath = anime_tsv_path()):
    
    tsv_data = pd.read_csv(animePath[0], sep = "\t") 
    
    for path in tqdm(animePath[1:]):
        tsv_data = tsv_data.append(pd.read_csv(path, sep ="\t"), ignore_index=True)
    
    with open("./generic_datafile/urls_anime.txt", "r") as file:
        lines = file.read().splitlines()
        tsv_data['url'] = np.resize(lines,len(tsv_data))

    tsv_data.to_csv("./generic_datafile/dataset.csv")
        
    return tsv_data


def read_dict(path = "./generic_datafile/inverted_index.txt"):
    
    dic = dict()
    
    with open(path, "r") as file:
        lines = file.read().splitlines()
        for line in lines:
            key = line.split(":")[0]
            values = line.split(":")[1].split(",")
            dic[key] = values

    return dic


def tfidf(i,j,d):
    """
    Input: (string, int, pandas.array)
    Output: int
    """
    
    ddad=[]

    for doc in d:
        if(isinstance(doc, str)):
            ddad.append(f.preprocessing(doc)) 
            #get all the sinonpsyses
            #in a list of set
    
    i=f.preprocessing(i)
    i=i.pop()
    
    n_ij=ddad[j].count(i)
    
   #tfij=n_ij/d_j where d_j=len(ddad[j])
    
    tfij=n_ij/len(ddad[j])
    
    idf_den=0

    for d in ddad:
        check=d.count(i)
        if check>0:
            idf_den+=1
    idf=np.log10(len(ddad)/idf_den)
    
    return tfij*idf

