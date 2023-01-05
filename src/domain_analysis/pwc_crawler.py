from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import json

base_url = "https://paperswithcode.com"
url = "https://paperswithcode.com/sota"

def get_top_categories():
    category_data = []

    page = requests.get(url)

    if page.status_code == 200:
        soup = BeautifulSoup(page.text, "html.parser")

    if soup:
        top_categories = soup.find_all("div", class_="row task-group-title")

        for category in top_categories:
            name = category.find("a").text
            href = category.find("a")["href"]
            
            data = {
                "category": name,
                "href": base_url + href
            }

            category_data.append(data)
        
    return category_data
        

def get_tasks(top_categories):
    
    for category in top_categories:
        tasks_data = set()
        sub_task_links = []
        page = requests.get(category["href"])

        if page.status_code == 200:
            soup = BeautifulSoup(page.text, "html.parser")

        if soup:
            container = soup.find("div", class_="infinite-container featured-task")
            tasks = container.find_all("h2")
            task_links = container.findAll("div", class_="sota-all-tasks")
                        
            for task in tasks:
                tasks_data.add(str(task.text).strip().split("(")[0])

            for task_link in task_links:
                href = task_link.find("a")
                if href: 
                    sub_task_links.append(base_url + href["href"])
                
        sub_task_data = get_sub_tasks(sub_task_links)

        tasks_data.update(sub_task_data)

        category["tasks"] = list(tasks_data)
        #category["sub_tasks"] = list(sub_task_data)

    return top_categories


def get_sub_tasks(links):

    sub_tasks_data = set()

    for link in links:
        page = requests.get(link)

        if page.status_code == 200:
            soup = BeautifulSoup(page.text, "html.parser")

        if soup:
            container = soup.find("div", class_="infinite-container featured-task area-tag-page")
            cards = container.findAll("div", class_="card")

            for card in cards:
                name = card.find("h1").text
                if name:
                    sub_tasks_data.add(name)

    return sub_tasks_data


def get_category_data():
    top_categories = get_top_categories()
    top_categories_with_tasks = get_tasks(top_categories)

    #print(type(top_categories_with_tasks))
    #print(top_categories_with_tasks)

    with open("categories_all.json", "w", encoding="utf-8") as dest:
        json.dump(top_categories_with_tasks, dest, sort_keys=True, indent=4)
    

def main():
    get_category_data()

if __name__ == "__main__":
    main()