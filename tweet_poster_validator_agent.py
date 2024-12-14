from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from langchain.agents import Tool
from langchain.agents.react.base import ReActDocstoreAgent
from langchain.tools.ddg_search import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from content_generation import ContentGenerator
from simulator import GreenPillBotTester
import logging
import json
from datetime import datetime

# State and action enums
class AgentState(str, Enum):
    GENERATE = "generate"
    VALIDATE = "validate"
    SEARCH = "search"
    POST = "post"
    END = "end"

class ActionType(str, Enum):
    CONTINUE = "continue"
    REGENERATE = "regenerate"
    VALIDATE = "validate"
    POST = "post"
    END = "end"

# State management
@dataclass
class TweetState:
    content: Optional[str] = None
    validation_results: List[Dict] = Field(default_factory=list)
    search_results: List[Dict] = Field(default_factory=list)
    posted: bool = False
    error: Optional[str] = None

class AgentState(BaseModel):
    tweet: TweetState = Field(default_factory=TweetState)
    history: List[Dict] = Field(default_factory=list)
    current_step: str = AgentState.GENERATE

class GreenPillAgent:
    def __init__(self):
        self.setup_logging()
        self.setup_components()
        self.setup_tools()
        self.setup_graph()

    def setup_logging(self):
        """Initialize logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def setup_components(self):
        """Initialize core components"""
        self.content_generator = ContentGenerator()
        self.bot = GreenPillBotTester()
        self.llm = ChatOpenAI(model="gpt-4")
        self.search = DuckDuckGoSearchRun()

    def setup_tools(self):
        """Initialize agent tools"""
        self.tools = [
            Tool(
                name="search",
                func=self.search.run,
                description="Search the web for information to validate tweet content"
            ),
            Tool(
                name="generate_tweet",
                func=self.generate_tweet,
                description="Generate a new tweet using the content generator"
            ),
            Tool(
                name="validate_tweet",
                func=self.validate_tweet,
                description="Validate tweet content for accuracy and appropriateness"
            ),
            Tool(
                name="post_tweet",
                func=self.post_tweet,
                description="Post the validated tweet"
            )
        ]

    def setup_graph(self):
        """Initialize the state graph"""
        self.graph = StateGraph(AgentState)
        
        # Add nodes
        self.graph.add_node(AgentState.GENERATE, self.generate_step)
        self.graph.add_node(AgentState.VALIDATE, self.validate_step)
        self.graph.add_node(AgentState.SEARCH, self.search_step)
        self.graph.add_node(AgentState.POST, self.post_step)
        
        # Add edges
        self.graph.add_edge(AgentState.GENERATE, AgentState.VALIDATE)
        self.graph.add_edge(AgentState.VALIDATE, AgentState.SEARCH)
        self.graph.add_edge(AgentState.SEARCH, AgentState.POST)
        self.graph.add_edge(AgentState.POST, END)
        
        # Add conditional edges for regeneration
        self.graph.add_conditional_edges(
            AgentState.VALIDATE,
            self.validate_router
        )
        
        self.graph.add_conditional_edges(
            AgentState.SEARCH,
            self.search_router
        )
        
        # Set entry point
        self.graph.set_entry_point(AgentState.GENERATE)

    async def generate_tweet(self) -> Dict:
        """Generate tweet content"""
        try:
            style = self.content_generator.get_tweet_style()
            content = self.bot.generate_tweet()
            
            return {
                "content": content,
                "style": style.name,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error generating tweet: {e}")
            return {"error": str(e)}

    async def validate_tweet(self, content: str) -> Dict:
        """Validate tweet content using LLM"""
        prompt = f"""Validate this tweet for accuracy and appropriateness:
        
        Tweet: {content}
        
        Check for:
        1. Technical accuracy
        2. Appropriate tone
        3. No misleading claims
        4. No sensitive content
        5. Proper context
        
        Return JSON with validation results."""
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = await chain.arun(content)
        
        return json.loads(result)

    async def search_validate(self, content: str) -> Dict:
        """Validate tweet content against search results"""
        try:
            # Extract key terms
            terms = self.extract_key_terms(content)
            
            # Search for each term
            search_results = []
            for term in terms:
                results = await self.search.arun(term)
                search_results.append({"term": term, "results": results})
            
            # Validate against search results
            validation = await self.llm.agenerate([{
                "role": "user",
                "content": f"""Validate tweet accuracy using search results:
                
                Tweet: {content}
                Search Results: {json.dumps(search_results)}
                
                Return JSON with validation assessment."""
            }])
            
            return json.loads(validation.generations[0].text)
            
        except Exception as e:
            self.logger.error(f"Error in search validation: {e}")
            return {"error": str(e)}

    def extract_key_terms(self, content: str) -> List[str]:
        """Extract key technical terms for validation"""
        chain = LLMChain(
            llm=self.llm,
            prompt="""Extract key technical terms from this tweet that should be validated:
            
            Tweet: {content}
            
            Return as JSON list."""
        )
        
        result = chain.run(content=content)
        return json.loads(result)

    async def generate_step(self, state: AgentState) -> AgentState:
        """Generate tweet step"""
        tweet_data = await self.generate_tweet()
        
        if "error" in tweet_data:
            state.tweet.error = tweet_data["error"]
            return state
            
        state.tweet.content = tweet_data["content"]
        state.history.append({
            "step": "generate",
            "action": "Generated initial tweet",
            "timestamp": datetime.now().isoformat()
        })
        
        return state

    async def validate_step(self, state: AgentState) -> AgentState:
        """Validate tweet step"""
        validation = await self.validate_tweet(state.tweet.content)
        state.tweet.validation_results.append(validation)
        
        state.history.append({
            "step": "validate",
            "action": "Performed content validation",
            "timestamp": datetime.now().isoformat()
        })
        
        return state

    async def search_step(self, state: AgentState) -> AgentState:
        """Search validation step"""
        search_validation = await self.search_validate(state.tweet.content)
        state.tweet.search_results.append(search_validation)
        
        state.history.append({
            "step": "search",
            "action": "Performed search validation",
            "timestamp": datetime.now().isoformat()
        })
        
        return state

    async def post_step(self, state: AgentState) -> AgentState:
        """Post tweet step"""
        try:
            response = self.bot.post_tweet()
            state.tweet.posted = True
            
            state.history.append({
                "step": "post",
                "action": "Posted tweet successfully",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            state.tweet.error = str(e)
            state.history.append({
                "step": "post",
                "action": f"Failed to post tweet: {e}",
                "timestamp": datetime.now().isoformat()
            })
            
        return state

    def validate_router(self, state: AgentState) -> str:
        """Route based on validation results"""
        if state.tweet.error:
            return AgentState.GENERATE
            
        validation = state.tweet.validation_results[-1]
        if validation.get("passed", False):
            return AgentState.SEARCH
        return AgentState.GENERATE

    def search_router(self, state: AgentState) -> str:
        """Route based on search validation results"""
        if state.tweet.error:
            return AgentState.GENERATE
            
        search_validation = state.tweet.search_results[-1]
        if search_validation.get("passed", False):
            return AgentState.POST
        return AgentState.GENERATE

    async def run(self) -> Dict:
        """Run the agent workflow"""
        try:
            state = AgentState()
            final_state = await self.graph.arun(state)
            
            return {
                "success": final_state.tweet.posted,
                "tweet": final_state.tweet.content,
                "history": final_state.history,
                "error": final_state.tweet.error
            }
            
        except Exception as e:
            self.logger.error(f"Agent workflow error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

if __name__ == "__main__":
    import asyncio
    
    async def main():
        agent = GreenPillAgent()
        result = await agent.run()
        print(json.dumps(result, indent=2))
    
    asyncio.run(main())