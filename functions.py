import sys
import requests
from bs4 import BeautifulSoup
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import pathlib
from pathlib import Path

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
    
    
def initGet(pageToGet = 400 ,CPUs = multiprocessing.cpu_count()):

    pages = [None] * pageToGet
    numberOfPage = range(0, pageToGet)

    pool = ThreadPool(CPUs)
    pool.map(lambda num : get_listAnimePage(num, pages), numberOfPage)   
    pool.map(lambda page : get_urls_In_ListAnimePage(page, pages), pages)
    
    return pages


def getAnime(pages, start=0):
    pages_from_start_to_end = pages[start:]
    for i in range(0, len(pages_from_start_to_end)) : 
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


def tdForCharacters_Voices(tag):
    return tag.name == "td" and not tag.has_attr("width") and tag.has_attr("valign")


def getDataFromPage(pagePath):
    with open(pagePath, "r") as file:
        soup = BeautifulSoup(file, "html.parser")

    temp = soup.find_all("div", {"class": "spaceit_pad"}) 

    out=[]
    tempDict = dict()
    finalDict = dict()

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

    if(tempDict["Episodes"] != "N/A"):
        finalDict["Episodes"] = int(tempDict["Episodes"]) #number of episodes
    else:
        finalDict["Episodes"] = None

    if(tempDict["Aired"] != "N/A"):
        finalDict["Aired"]  = tempDict["Aired"]#releasedDate and endDate
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

    finalDict["Synopsis"] = temp.text.replace("\n", " ")

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
