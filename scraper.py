import feedparser

def get_google_news(query="stock"):
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)

    # Collect all titles into a list
    titles = [entry.title for entry in feed.entries]

    return titles

def get_yahoo_news(query="stock"):
    rss_url = f"https://news.yahoo.com/rss/{query}"
    feed = feedparser.parse(rss_url)

    # Collect all titles into a list
    titles = [entry.title for entry in feed.entries]

    return titles

def get_reddit_posts(subreddit="wallstreetbets"):
    feed = feedparser.parse(f"https://www.reddit.com/r/{subreddit}.rss")

    # Collect all titles into a list
    titles = [entry.title for entry in feed.entries]

    return titles

def get_tweets(query="stock"):
    pass