import json
import wget
from typing import List
import subprocess

def get_paper_data() -> List:
    paper_data = []
    
    with open("final_metadata_set.json", "r", encoding="utf-8") as src:
        data = json.load(src)


    for item in data:
        papers = item["papers"][0]
        title = papers["paper_title"]
        url = papers["paper_url_pdf"]

        
        paper_data.append((title, url))


    return paper_data




def download_pdfs(paper_data: List) -> None:
    counter = 0
    for _, link in paper_data:
        try:
            counter += 1
            wget.download(link, out="D:\\AgileAI\\Parameter Settings\\Paper")
            print(counter)
        except:
            pass

        
if __name__ == "__main__":
    paper_data = get_paper_data()
    download_pdfs(paper_data)