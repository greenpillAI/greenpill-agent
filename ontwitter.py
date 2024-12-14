from typing import List, Dict, Optional
import tweepy
import logging
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from greenpill_bot import GreenPillBot
from typing import List, Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from langchain.agents import Tool
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import tweepy
import logging
from datetime import datetime
import json
from content_generation import ContentGenerator
from duckduckgo_search import ddg


@dataclass
class TweetContext:
    tweet_id: str
    text: str
    author_id: str
    conversation_id: str
    created_at: datetime
    in_reply_to_id: Optional[str] = None
    author_username: Optional[str] = None

from typing import List, Dict, Optional
import tweepy
import logging
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from simulator import GreenPillBotTester  # Change this import

class TwitterHandler:
    def __init__(self):
        load_dotenv()
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self._setup_twitter()
        self.bot_username = "greenpillai"
        self.last_mention_id = None
        self.processed_tweets = set()
        self.bot = GreenPillBotTester()  # Use the tester class instead
        self._initialize_mention_tracker()

    def _setup_twitter(self):
        """Initialize Twitter API client with error handling"""
        required_env_vars = [
            "TWITTER_BEARER_TOKEN",
            "TWITTER_API_KEY",
            "TWITTER_API_SECRET",
            "TWITTER_ACCESS_TOKEN",
            "TWITTER_ACCESS_SECRET"
        ]
        
        # Check for missing environment variables
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
        try:
            self.client = tweepy.Client(
                bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
                consumer_key=os.getenv("TWITTER_API_KEY"),
                consumer_secret=os.getenv("TWITTER_API_SECRET"),
                access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
                access_token_secret=os.getenv("TWITTER_ACCESS_SECRET"),
                wait_on_rate_limit=True
            )
            # Verify credentials
            self.client.get_me()
            self.logger.info("Twitter authentication successful")
        except Exception as e:
            self.logger.error(f"Twitter setup error: {e}")
            raise

    def _initialize_mention_tracker(self):
        """Initialize the last mention ID to only process new mentions"""
        try:
            # Get the most recent mention to establish starting point
            recent_mentions = self.client.get_users_mentions(
                self.client.get_me().data.id,
                tweet_fields=["created_at"],
                max_results=5
            )

            if recent_mentions.data:
                self.last_mention_id = max(mention.id for mention in recent_mentions.data)
                self.logger.info(f"Initialized last_mention_id to {self.last_mention_id}")
                
                for mention in recent_mentions.data:
                    self.processed_tweets.add(mention.id)
            else:
                self.logger.info("No existing mentions found")

        except Exception as e:
            self.logger.error(f"Error initializing mention tracker: {e}")
            raise

    def get_tweet_context(self, tweet_id: str) -> List[TweetContext]:
        """Get the full conversation context for a tweet"""
        try:
            # Get the tweet and its data
            tweet = self.client.get_tweet(
                tweet_id,
                expansions=["author_id", "referenced_tweets.id", "in_reply_to_user_id"],
                tweet_fields=["created_at", "conversation_id", "author_id"],
                user_fields=["username"]
            )
            
            if not tweet.data:
                return []

            conversation_id = tweet.data.conversation_id
            context_tweets = []

            # Get conversation tweets
            conversation = self.client.search_recent_tweets(
                query=f"conversation_id:{conversation_id}",
                expansions=["author_id", "referenced_tweets.id"],
                tweet_fields=["created_at", "in_reply_to_user_id", "author_id"],
                user_fields=["username"],
                max_results=100
            )

            if conversation.data:
                # Create user lookup dictionary
                users = {user.id: user.username for user in conversation.includes['users']} if 'users' in conversation.includes else {}
                
                # Sort tweets by creation time
                tweets = sorted(conversation.data, key=lambda x: x.created_at)
                
                for t in tweets:
                    context_tweets.append(
                        TweetContext(
                            tweet_id=t.id,
                            text=t.text,
                            author_id=t.author_id,
                            author_username=users.get(t.author_id),
                            conversation_id=conversation_id,
                            created_at=t.created_at,
                            in_reply_to_id=getattr(t, "in_reply_to_user_id", None)
                        )
                    )

            return context_tweets

        except Exception as e:
            self.logger.error(f"Error getting tweet context: {e}")
            return []

    def process_mentions(self):
        """Process only new mentions and generate replies"""
        try:
            # Get mentions newer than our last processed mention
            mentions = self.client.get_users_mentions(
                self.client.get_me().data.id,
                expansions=["author_id", "referenced_tweets.id"],
                tweet_fields=["created_at", "conversation_id", "author_id"],
                user_fields=["username"],
                since_id=self.last_mention_id,
                max_results=100
            )

            if not mentions.data:
                self.logger.info("No new mentions found")
                return

            for mention in mentions.data:
                if mention.id in self.processed_tweets:
                    continue

                try:
                    self.logger.info(f"Processing new mention: {mention.id}")
                    context = self.get_tweet_context(mention.id)
                    self._handle_mention(mention, context)
                    
                    self.processed_tweets.add(mention.id)
                    self.last_mention_id = max(mention.id, self.last_mention_id) if self.last_mention_id else mention.id
                    
                    # Respect rate limits
                    time.sleep(2)

                except Exception as e:
                    self.logger.error(f"Error processing mention {mention.id}: {e}")

        except Exception as e:
            self.logger.error(f"Error in process_mentions: {e}")

    def _clean_mention(self, mention_text: str) -> str:
        """Remove bot's handle and clean up the mention text"""
        # Remove @greenpillai (case insensitive)
        cleaned_text = mention_text.replace(f"@{self.bot_username}", "").strip()
        # Remove any leading/trailing whitespace and handle multiple spaces
        cleaned_text = " ".join(cleaned_text.split())
        return cleaned_text

    def _handle_mention(self, mention: tweepy.Tweet, context: List[TweetContext]):
        """Handle individual mentions with context"""
        try:
            # Clean the mention text before processing
            cleaned_mention = self._clean_mention(mention.text)
            conversation_text = self._format_conversation(context)
            
            # Use cleaned mention text for generating reply
            reply = self.bot.test_reply(cleaned_mention)
            
            if reply and reply.response:  # Check both reply object and its response
                response = self.client.create_tweet(
                    text=reply.response,
                    in_reply_to_tweet_id=mention.id
                )
                self.logger.info(f"Posted reply to tweet {mention.id}: {reply.response[:50]}...")
                return response
            else:
                self.logger.info(f"No reply generated for mention {mention.id} - not relevant to bot's purpose")
                
        except Exception as e:
            self.logger.error(f"Error handling mention: {e}")

    def _format_conversation(self, context: List[TweetContext]) -> str:
        """Format conversation context for the reply generator"""
        formatted = []
        for tweet in context:
            author = f"@{tweet.author_username}" if tweet.author_username else f"user_{tweet.author_id}"
            formatted.append(f"{author}: {tweet.text}")
        return "\n".join(formatted)

    def _generate_reply(self, mention_text: str, context: str) -> Optional[str]:
        """Generate reply using the bot's reply generation method"""
        try:
            prompt = f"Conversation context:\n{context}\n\nMention: {mention_text}"
            reply = self.bot.generate_reply(prompt)
            
            # Ensure reply fits Twitter's character limit
            if reply and len(reply) > 280:
                reply = reply[:277] + "..."
                
            return reply
        except Exception as e:
            self.logger.error(f"Error generating reply: {e}")
            return None

    def run(self, interval_minutes: int = 3):
        """Run the mention processor at regular intervals"""
        self.logger.info(f"Starting Twitter handler, checking for new mentions every {interval_minutes} minutes")
        self.logger.info(f"Initial last_mention_id: {self.last_mention_id}")
        
        while True:
            try:
                self.process_mentions()
                time.sleep(interval_minutes * 60)
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error




class ReplyState(str, Enum):
    ANALYZE = "analyze"
    VALIDATE = "validate"
    SEARCH = "search"
    GENERATE = "generate"
    POST = "post"
    END = "end"

@dataclass
class ConversationContext:
    tweets: List[TweetContext] = field(default_factory=list)
    sentiment: Optional[str] = None
    key_topics: List[str] = field(default_factory=list)
    technical_terms: List[str] = field(default_factory=list)
    search_results: Dict = field(default_factory=dict)
    
@dataclass
class ReplyData:
    original_tweet: TweetContext
    context: ConversationContext
    generated_reply: Optional[str] = None
    validation_results: List[Dict] = field(default_factory=list)
    error: Optional[str] = None
    posted: bool = False

class AgentState(BaseModel):
    reply: ReplyData = Field(...)
    history: List[Dict] = Field(default_factory=list)
    current_step: str = ReplyState.ANALYZE

class TwitterReplyAgent:
    def __init__(self, twitter_handler: TwitterHandler):
        self.twitter = twitter_handler
        self.setup_logging()
        self.setup_components()
        self.setup_tools()
        self.setup_graph()

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def setup_components(self):
        """Initialize core components"""
        self.content_generator = ContentGenerator()
        self.llm = ChatOpenAI(model="gpt-4")
        
    def setup_tools(self):
        """Initialize agent tools"""
        self.tools = [
            Tool(
                name="analyze_conversation",
                func=self.analyze_conversation,
                description="Analyze conversation context and sentiment"
            ),
            Tool(
                name="search_context",
                func=self.search_context,
                description="Search for additional context about technical terms"
            ),
            Tool(
                name="validate_reply",
                func=self.validate_reply,
                description="Validate generated reply for accuracy and tone"
            ),
            Tool(
                name="generate_reply",
                func=self.generate_reply,
                description="Generate contextual reply"
            )
        ]

    def setup_graph(self):
        """Initialize the state graph"""
        self.graph = StateGraph(AgentState)
        
        # Add nodes
        self.graph.add_node(ReplyState.ANALYZE, self.analyze_step)
        self.graph.add_node(ReplyState.SEARCH, self.search_step)
        self.graph.add_node(ReplyState.GENERATE, self.generate_step)
        self.graph.add_node(ReplyState.VALIDATE, self.validate_step)
        self.graph.add_node(ReplyState.POST, self.post_step)
        
        # Add edges
        self.graph.add_edge(ReplyState.ANALYZE, ReplyState.SEARCH)
        self.graph.add_edge(ReplyState.SEARCH, ReplyState.GENERATE)
        self.graph.add_edge(ReplyState.GENERATE, ReplyState.VALIDATE)
        self.graph.add_edge(ReplyState.VALIDATE, ReplyState.POST)
        self.graph.add_edge(ReplyState.POST, END)
        
        # Add conditional edges
        self.graph.add_conditional_edges(
            ReplyState.VALIDATE,
            self.validate_router
        )
        
        self.graph.add_conditional_edges(
            ReplyState.GENERATE,
            self.generate_router
        )
        
        # Set entry point
        self.graph.set_entry_point(ReplyState.ANALYZE)

    async def analyze_conversation(self, context: List[TweetContext]) -> Dict:
        """Analyze conversation context using LLM"""
        conversation_text = self.twitter._format_conversation(context)
        
        prompt = f"""Analyze this conversation context:

        {conversation_text}

        Provide:
        1. Overall sentiment
        2. Key discussion topics
        3. Technical terms used
        4. Conversation style
        
        Return as JSON."""
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = await chain.arun(conversation_text)
        return json.loads(result)

    async def search_context(self, terms: List[str]) -> Dict[str, List[str]]:
        """Search for technical terms context"""
        results = {}
        for term in terms:
            try:
                search_results = ddg(term, max_results=3)
                results[term] = [result['link'] for result in search_results]
            except Exception as e:
                self.logger.error(f"Search error for term {term}: {e}")
                results[term] = []
        return results

    async def validate_reply(self, reply: str, context: ConversationContext) -> Dict:
        """Validate reply content"""
        prompt = f"""Validate this reply in context:

        Original Context: {context.tweets[-1].text}
        Generated Reply: {reply}

        Check for:
        1. Technical accuracy (using search results)
        2. Tone appropriateness
        3. Context relevance
        4. GreenPill alignment
        
        Return JSON with validation results and score (0-1)."""
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = await chain.arun(reply)
        return json.loads(result)

    async def generate_reply(self, state: ReplyData) -> str:
        """Generate contextual reply"""
        style = self.content_generator.get_reply_style()
        
        # Create comprehensive prompt
        prompt = f"""Generate a reply with this context:

        Conversation: {self.twitter._format_conversation(state.context.tweets)}
        Sentiment: {state.context.sentiment}
        Key Topics: {', '.join(state.context.key_topics)}
        Technical Terms: {', '.join(state.context.technical_terms)}
        
        Style Guidelines:
        {style.template}
        
        Additional Context:
        {json.dumps(state.context.search_results, indent=2)}
        
        Generate a technically accurate, engaging reply under 280 characters."""
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return await chain.arun()

    async def analyze_step(self, state: AgentState) -> AgentState:
        """Analyze conversation step"""
        try:
            analysis = await self.analyze_conversation(state.reply.context.tweets)
            
            state.reply.context.sentiment = analysis.get('sentiment')
            state.reply.context.key_topics = analysis.get('topics', [])
            state.reply.context.technical_terms = analysis.get('technical_terms', [])
            
            state.history.append({
                "step": "analyze",
                "result": "Completed conversation analysis",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            state.reply.error = f"Analysis error: {str(e)}"
            
        return state

    async def search_step(self, state: AgentState) -> AgentState:
        """Search context step"""
        try:
            if state.reply.context.technical_terms:
                search_results = await self.search_context(
                    state.reply.context.technical_terms
                )
                state.reply.context.search_results = search_results
                
            state.history.append({
                "step": "search",
                "result": f"Searched {len(state.reply.context.technical_terms)} terms",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            state.reply.error = f"Search error: {str(e)}"
            
        return state

    async def generate_step(self, state: AgentState) -> AgentState:
        """Generate reply step"""
        try:
            reply = await self.generate_reply(state.reply)
            state.reply.generated_reply = reply
            
            state.history.append({
                "step": "generate",
                "result": "Generated reply",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            state.reply.error = f"Generation error: {str(e)}"
            
        return state

    async def validate_step(self, state: AgentState) -> AgentState:
        """Validate reply step"""
        try:
            validation = await self.validate_reply(
                state.reply.generated_reply,
                state.reply.context
            )
            state.reply.validation_results.append(validation)
            
            state.history.append({
                "step": "validate",
                "result": f"Validation score: {validation.get('score', 0)}",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            state.reply.error = f"Validation error: {str(e)}"
            
        return state

    async def post_step(self, state: AgentState) -> AgentState:
        """Post reply step"""
        try:
            if not state.reply.error and state.reply.generated_reply:
                response = self.twitter.client.create_tweet(
                    text=state.reply.generated_reply,
                    in_reply_to_tweet_id=state.reply.original_tweet.tweet_id
                )
                state.reply.posted = True
                
                state.history.append({
                    "step": "post",
                    "result": "Posted reply successfully",
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            state.reply.error = f"Posting error: {str(e)}"
            state.history.append({
                "step": "post",
                "result": f"Failed to post: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
            
        return state

    def validate_router(self, state: AgentState) -> str:
        """Route based on validation results"""
        if state.reply.error:
            return ReplyState.GENERATE
            
        latest_validation = state.reply.validation_results[-1]
        if latest_validation.get('score', 0) >= 0.8:
            return ReplyState.POST
        return ReplyState.GENERATE

    def generate_router(self, state: AgentState) -> str:
        """Route based on generation results"""
        if state.reply.error or not state.reply.generated_reply:
            if len(state.history) < 3:  # Limit retries
                return ReplyState.GENERATE
            return END
        return ReplyState.VALIDATE

    async def process_mention(self, mention: tweepy.Tweet, context: List[TweetContext]) -> Dict:
        """Process a single mention"""
        try:
            # Initialize state
            initial_state = AgentState(
                reply=ReplyData(
                    original_tweet=mention,
                    context=ConversationContext(tweets=context)
                )
            )
            
            # Run the graph
            final_state = await self.graph.arun(initial_state)
            
            return {
                "success": final_state.reply.posted,
                "reply": final_state.reply.generated_reply,
                "history": final_state.history,
                "error": final_state.reply.error
            }
            
        except Exception as e:
            self.logger.error(f"Error processing mention: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Modify TwitterHandler to use the new agent
class EnhancedTwitterHandler(TwitterHandler):
    def __init__(self):
        super().__init__()
        self.reply_agent = TwitterReplyAgent(self)
        
    async def _handle_mention(self, mention: tweepy.Tweet, context: List[TweetContext]):
        """Enhanced mention handling using LangGraph agent"""
        try:
            result = await self.reply_agent.process_mention(mention, context)
            
            if result["success"]:
                self.logger.info(f"Successfully processed mention {mention.id}")
            else:
                self.logger.error(f"Failed to process mention {mention.id}: {result.get('error')}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error in enhanced mention handling: {e}")
            return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import asyncio
    
    async def main():
        handler = EnhancedTwitterHandler()
        await handler.run()
    
    asyncio.run(main())