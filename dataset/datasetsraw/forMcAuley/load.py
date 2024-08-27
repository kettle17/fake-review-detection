from datasets import load_dataset
import datetime
import csv
import random
import sys


loopnum = 200 #number of articles to get from dataset total
searchnum = 99999 #number of reviews+1 to search the start of the dataset for before randomising

#the dataset to grab from text. all are structured the same
datasetcat = []
with open("all_categories.txt", "r") as fileobj:
    for x in fileobj:
        datasetcat.append(x.rstrip('\n'))

unixtime = 1648771200000 #timestamp to get reviews from afterwards

with open('FRDDS.csv', 'w', newline='', encoding="utf-8-sig") as file:
    
    #create csv with same name.
    writer = csv.writer(file)

    #all headers of parameters taken from the two datasets. 1st and 3rd row are from review and meta respectively. 'parent_asin' is in both.
    headers = ["title", "text", "rating", "category", "label", "images", "timestamp",
                "parent_asin",
                'item_title']
        
    #first line of csv writes the headers
    writer.writerow(headers)
    
    #loop through each dataset and grab articles
    for datasetgrab in datasetcat:

        #init in loop to reset each new category
        firstentries = []
        newentries = []
        #load meta+review dataset. we will be partially merging both
        dataset = load_dataset("Amazon-Reviews-2023", "raw_review_" + datasetgrab, trust_remote_code=True)
        meta_dataset = load_dataset("Amazon-Reviews-2023", "raw_meta_" + datasetgrab, split="full", trust_remote_code=True)

        
        
        #first loop: search start of dataset to the searchnum cutoff
        for entry in dataset["full"]:
            if (entry["timestamp"] > unixtime): #if the review is past the timestamp, add it
                if (len(firstentries)>searchnum):
                    break

                imagefield = "" #default image value is [] so set default to null for reading later

                if entry["images"]:
                    imagefield = entry["images"][0]["large_image_url"] #if there is value its a dict. we only need the url, and not really even that. url is important for verif reasons though

                unixtimealt = int(entry["timestamp"]) / 1000 #convert unixtime to datetime for insertion
                value = datetime.datetime.fromtimestamp(unixtimealt)

                #insert entries into the first list
                firstentries.append([entry["title"],entry["text"],entry["rating"],datasetgrab,"OR",
                                    imagefield, f"{value:%H:%M:%S %d-%m-%Y}",entry["parent_asin"]]) 
                print("appended")
                print(entry["timestamp"])
        print(len(firstentries))

        #setup for 2nd list
        for i in range(loopnum):
            gennumber = random.randint(0,len(firstentries)) #random number between 0 and the length of list. get value at that index. allows for not just doing the earliest entries each time
            currentry = firstentries[gennumber]  
            newentries.append(currentry) #append to 2nd list, which we will now iterate through to insert the meta entries
        

        #loop through meta dataset to find shared asin ID. if there is a match then append metadataset values to that entry. allows for merge
        count = 0
        fillcount = len(newentries)
        for meta_entry in meta_dataset:
            print("\ritem", count, "of", len(meta_dataset))
            sys.stdout.write("\033[F")
            for entry in newentries:
                if (meta_entry["parent_asin"] == entry[7]):
                    print("found item")
                    entry.append(meta_entry["title"])
                    fillcount = fillcount - 1
                    break
            count = count+1
            if fillcount <= 0:
                break
        
        #write entry to csv
        for entry in newentries:
            writer.writerow(entry)
        
        #write an additional 200 rows of blank data containing only category and label, to be filled in by AI
        for i in range(loopnum):
            writer.writerow(["","","",datasetgrab,"CG"])
        