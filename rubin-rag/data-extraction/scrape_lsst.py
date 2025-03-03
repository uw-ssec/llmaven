import requests
import pandas as pd
import time
from tqdm import tqdm

BASE_URL = "https://community.lsst.org"
LATEST_URL = f"{BASE_URL}/latest.json"  # to get the latest topics and posts first, use '/latest.json' endpoint.

def fetch_topics(page):
    """
    Fetch topics from the LSST forum.
    Args:
        page (int): The page number to fetch.
    Returns:
        dict: JSON response containing topics data or None if request fails.
    """
    url = f"{LATEST_URL}?page={page}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # raise error if response is not 200
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page {page}: {e}")
        return None

def fetch_topic_details(topic_id):
    """
    Fetch full details of a specific topic using its topic ID.
    Args:
        topic_id (int): The topic's unique ID.
    Returns:
        dict: JSON response containing topic details or None if request fails.
    """
    url = f"{BASE_URL}/t/{topic_id}.json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching topic {topic_id}: {e}")
        return None

def extract_topic_data(topic):
    """
    Extract topic details such as question, author, category, views, etc.
    Args:
        topic (dict): JSON data of a topic.
    Returns:
        dict: Extracted topic metadata.
    """
    return {
        "topic_id": topic.get("id"),
        "question": topic.get("title"),
        "question_author": topic.get("posters")[0]["user_id"] if topic.get("posters") else None,
        "question_date": topic.get("created_at"),
        "category_id": topic.get("category_id"),
        "views": topic.get("views"),
        "posts_count": topic.get("posts_count"),
        "last_activity": topic.get("bumped_at")
    }

def extract_reply_data(post, topic_id):
    """
    Extract reply details including answer text, author, moderator/admin status.
    Args:
        post (dict): JSON data of a post (reply).
        topic_id (int): The topic ID the reply belongs to.
    Returns:
        dict: Extracted reply metadata.
    """
    return {
        "topic_id": topic_id,
        "answer": post.get("cooked"),
        "answer_author": post.get("username"),
        "answer_date": post.get("created_at"),
        "primary_group_name": post.get("primary_group_name"),
        "flair_name": post.get("flair_name"),
        "flair_url": post.get("flair_url"),
        "moderator": post.get("moderator"),
        "admin": post.get("admin"),
        "staff": post.get("staff"),
        "is_accepted_answer": post.get("accepted_answer", False)
    }

def scrape_forum(delay=2):
    """
    Scrape all topics and replies, Store extracted data.
    Args:
        delay (int, optional): Delay between requests to prevent server overload. Defaults to 2 seconds.
    Returns:
        list: List of extracted question-answer pairs.
    """
    qa_pairs = []
    page = 1  # there is something similar to pageniation even if its not explitely mentioned - start from the first page

    while True:
        print(f"Fetching page {page}...")
        data = fetch_topics(page)
        if not data:
            break  # come out of the loop if API request fails

        topics = data.get("topic_list", {}).get("topics", [])
        if not topics:
            print("No more topics found. Stopping.")  # come out of the loop if no more topics are found
            break  

        for topic in tqdm(topics, desc=f"Processing Page {page}"):
            topic_data = extract_topic_data(topic)  # extract metadata for the topic

            # fetch topic details including all replies
            topic_details = fetch_topic_details(topic_data["topic_id"])
            if topic_details:
                posts = topic_details.get("post_stream", {}).get("posts", [])  # get all posts in the topic
                for post in posts[1:]:  # skip first post (it's the main topic)
                    reply_data = extract_reply_data(post, topic_data["topic_id"])  # extract metadata for each reply

                    # keep appending to the dictionary
                    qa_pairs.append({
                        "category_id": topic_data["category_id"],
                        "question": topic_data["question"],
                        "question_author": topic_data["question_author"],
                        "question_date": topic_data["question_date"],
                        "answer": reply_data["answer"],
                        "answer_author": reply_data["answer_author"],
                        "answer_date": reply_data["answer_date"],
                        "primary_group_name": reply_data["primary_group_name"],
                        "flair_name": reply_data["flair_name"],
                        "flair_url": reply_data["flair_url"],
                        "moderator": reply_data["moderator"],
                        "admin": reply_data["admin"],
                        "staff": reply_data["staff"],
                        "is_accepted_answer": reply_data["is_accepted_answer"]
                    })

            time.sleep(delay)

        page += 1  # move to the next page

    return qa_pairs

def main():
    """
    Main function to execute the scraping process.
    """
    # scrape forum data (topics, replies, and metadata)
    qa_data = scrape_forum()

    qa_df = pd.DataFrame(qa_data)
    qa_df["question_date"] = pd.to_datetime(qa_df["question_date"]).dt.date
    qa_df["answer_date"] = pd.to_datetime(qa_df["answer_date"]).dt.date
    qa_df.sort_values(by=["question_date", "answer_date"], ascending=[False, False], inplace=True)
    qa_df.reset_index(drop=True, inplace=True)

    # save the data to a CSV file
    csv_filename = "scraped_data/lsst_forum_responses.csv"
    qa_df.to_csv(csv_filename, index=False)
    print("Scraping complete! Data successfully saved.")

