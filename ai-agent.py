# %%
# Libs needed: can be added to requirements.txt
# pip install google-genai
# pip install python-dotenv
# pip install google-adk

# %%
import os
from dotenv import load_dotenv
from google import genai
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
from datetime import date
import textwrap
from IPython.display import display, Markdown
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

# Carrega a API_KEY (insira no arquivo .env)
GEMINI_API_KEY = os.environ.get("API_KEY")
if not GEMINI_API_KEY:
    print("API_KEY not found. Verify your .env file.")
    exit()

# %%

# Gera o cliente
client = genai.Client(api_key=GEMINI_API_KEY)

# Gera o modelo
model = "gemini-2.0-flash"

# %%
def call_agent(agent: Agent, message_text: str) -> str:
    session_service = InMemorySessionService().create_session(
        app_name=agent.name, 
        user_id="user1", 
        session_id="session1"
    )
    runner = Runner(
        agent=agent, 
        app_name=agent.name, 
        session_service=session_service
    )
    content = types.Content(
        role="user", 
        parts=[types.Part(text=message_text)]
    )

    final_response = ""

    for event in runner.run(user_id="user1", session_id="session1", new_message=content):
        if event.is_final_response():
            for part in event.content.parts:
                final_response += part.text
                final_response += "\n"
    return final_response.strip()

# %%
def to_markdown(text: str):
    text = text.replace('•', '*')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# %%
def search_agent(topic, today_date):
    search = Agent(
        name="Search Agent",
        model=model,
        description="AI Agent that searches the web for information.",
        tools=[google_search],
        instruction=""" 
        You are a search agent. Your job is to search the web using google_search tool to retreive the latest news about the topic {topic}.
        Focus on the five most relevant results, based on how hot the topic is based on the date.
        If a topic is not hot, you should discard it.
        You will only return results within the last 30 days.
        """
    )
    search_input = f"Topic: {topic}\nDate: {today_date}\n"
    first_agent = call_agent(search, search_input)
    return first_agent

# %%
def planner_agent(topic, news):
    planner = Agent(
        name="Planner Agent",
        model=model,
        description="AI Agent that creates a posts planner for the related news",
        tools=[google_search],
        instruction="""
        You are a social media content planner agent. Your job is to create social media content based on the news {news}.
        You will use google_search tool to gather which social media platforms are the best to post the content.
        You will select the most relevant news and return it with a plan with the subjects to be posted.
        """
    )
    planner_input = f"topic: {topic}\nNews: {news}\n"
    second_agent = call_agent(planner, planner_input)
    return second_agent

# %%
def writer_agent(topic, plan):
    writer = Agent(
        name="Writer Agent",
        model=model,
        description="AI Agent responsible for writing the content based on the plan provided",
        instruction="""
        You are a creative and modern social media content copywriter. Your specialty is to write contents that go viral in every social media platform.
        You will use your knowledge and creativity to write the content based on the plan {plan}, and with specific writing styles for each social media platform.
        You will use the plan provided to create your writing.
        The post MUST be highly engaging, informaative, with proper language and include at least 2 hashtags related to the topic.
        """
    )
    writer_input = f"topic: {topic}\nPost: {plan}\n"
    third_agent = call_agent(writer, writer_input)
    return third_agent

# %%
def reviewer_agent(topic, post):
    reviewer = Agent(
        name="Reviewer Agent",
        model=model,
        description="Reviewer Agent responsible for overviewing the content provided by previous agents",
        instruction="""
        You are a thorough and detail-oriented content reviewer. You are able to catch things that others miss. You are one of the best at what you do.
        You will use your knowledge and experience to review the content {post}.
        The original topic is {topic}.
        You will check for grammar, spelling and punctuation errors. Also, you will check if the content is engaging and informative. You will also compare the content with the topic provided and check if it is related to the topic. You will als compare with similar content and check if it is original.
        You will block any content that is not related to the topic, not original or that has pornography, hate speech, curse words or any other kind of content that is over PG-13.
        Any mistakes found should be reported and highlighted for improvement.
        """
    )
    reviewer_input = f"topic: {topic}\nPost: {post}\n"
    fourth_agent = call_agent(reviewer, reviewer_input)
    return fourth_agent

# %%
def run_agent_pipeline(topic: str):
    print(f"\n🚀 Initializing agents pipeline for the topic: '{topic}'")
    today_str = date.today().strftime("%d/%m/%Y")

    print("\n1. Searching for news...")
    news_summary = search_agent(topic, today_str)
    if not news_summary or "discard it" in news_summary.lower() or "no relevant results" in news_summary.lower():
        print("🔍 Search agent did not found any relevant news of the regarding topic. Exiting pipeline...")
        return "Not a single relevant news was found."
    print("📰 News found:\n", to_markdown(news_summary))

    print("\n2. Creating post planner...")
    content_plan = planner_agent(topic, news_summary)
    print("📝 Content Planner:\n", to_markdown(content_plan))

    print("\n3. Writing Agent creating copy...")
    draft_posts = writer_agent(topic, content_plan)
    print("✍️ Drafts:\n", to_markdown(draft_posts))

    print("\n4. Reviewing agent analyzing posts...")
    final_reviewed_post = reviewer_agent(topic, draft_posts)
    print("\n✨ Content is now reviewed ✨\n")
    print(to_markdown(final_reviewed_post)) 

    return final_reviewed_post

# %%
def chatbot():
    print("🤖 Hi! I'm your AI social media assistant for creating social media posts.")
    print("   To generate a post content, type: generate post about: <your topic here>")
    print("   To exit, type: exit")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "exit":
            print("🤖 See you soon!")
            break
        elif user_input.lower().startswith("generate post about:"):
            topic = user_input.lower().replace("generate post about:", "").strip()
            if not topic:
                print("🤖 Please, specify a topic after you type 'generate post about:'.")
            else:
                run_agent_pipeline(topic)
        else:
            print("🤖 Unknown command line. Try something like 'generate post about: <topic>' or 'exit'.")

# %%
if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("ERROR: The ENV VAR API_KEY is not defined or is INVALID.")
        print("Please, create a .env file with API_KEY='your_api_key_here' or define a Environment Variable.")
    else:
        chatbot()

# %%



