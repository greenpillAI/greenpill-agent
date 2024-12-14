from greenpill_bot import GreenPillBot
from content_generation import ContentGenerator
import logging
from typing import List, Dict, Optional
from datetime import datetime
import json
import os
from dataclasses import dataclass, asdict
import time
from rich.console import Console
from rich.table import Table
from rich import print as rprint

@dataclass
class TestResult:
    timestamp: str
    type: str
    content: str
    char_count: int
    is_relevant: Optional[bool] = None
    response: Optional[str] = None
    error: Optional[str] = None

class GreenPillBotTester(GreenPillBot):
    def __init__(self):
        super().__init__()
        self._setup_expanded_topics()
        self._setup_relevance_checker()
        self.content_generator = ContentGenerator()
        self.test_results: List[TestResult] = []
        self.console = Console()

    def _setup_expanded_topics(self):
        """Extended topic list for comprehensive testing"""
        additional_topics = [
            # Core GreenPill Concepts
            "token engineering for good",
            "circular economy blockchain",
            "impact investing web3",
            "quadratic funding implementation",
            "retroactive public goods",
            "regenerative cryptoeconomics",
            "solarpunk technology",
            
            # Economic Models
            "token weighted voting",
            "conviction voting systems",
            "bonding curves design",
            "curation markets",
            "impact certificates",
            "proof of impact metrics",
            
            # Environmental Focus
            "regenerative finance",
            "carbon credit tokens",
            "climate DAOs",
            "environmental impact tracking",
            "sustainable blockchain scaling",
            "green consensus mechanisms",
            
            # Governance Innovation
            "decentralized governance research",
            "optimistic governance",
            "futarchy implementations",
            "quadratic voting systems",
            "governance minimization",
            "progressive decentralization",
            
            # Social Impact
            "community coordination tools",
            "public goods funding",
            "impact DAOs",
            "social token design",
            "community ownership",
            "stakeholder capitalism",
            
            # Technical Implementation
            "mechanism design patterns",
            "governance frameworks",
            "smart contract coordination",
            "token engineering tools",
            "on-chain governance",
            "blockchain coordination games"
        ]
        self.topics.extend(additional_topics)
        self.logger.info(f"Extended topics list with {len(additional_topics)} new topics")

    def _setup_relevance_checker(self):
      """Initialize comprehensive list of relevant topics"""
      self.relevant_topics = self.topics + [
          # Core GreenPill and Regen Concepts
          "regenerative crypto",
          "regenerative cryptocurrency",
          "regen crypto",
          "greenpill",
          "green pill",
          "greenpill network",
          "greenpill book",
          "regenerative cryptoeconomics",
          "regenerative finance",
          "regen finance",
          "impact certificates",
          "proof of impact",
          
          # Key People and Projects
          "kevin owocki",
          "gitcoin",
          "coordinate colorado",
          "protocol guild",
          "optimism",
          "gitcoin grants",
          "zcash major grants",
          "opensearch dao",
          
          # Economic and Funding Mechanisms
          "retroactive funding",
          "impact market design",
          "quadratic funding",
          "harberger taxes",
          "token curated registries",
          "conviction voting",
          "rage quitting",
          "bonding curves",
          "moloch dao",
          "decentralized funding mechanisms",
          "public goods funding",
          "token weighted voting",
          
          # Governance and Coordination
          "mechanism design",
          "cryptoeconomics",
          "protocol governance",
          "progressive decentralization",
          "blockchain coordination",
          "optimistic governance",
          "blockchain governance experiments",
          "schelling points",
          "coordination mechanisms",
          "governance minimization",
          "governance frameworks",
          
          # Technical Concepts
          "optimistic rollups for good",
          "network states",
          "smart contract coordination",
          "on-chain governance",
          "mechanism design patterns",
          "token engineering",
          "cryptoeconomic primitives",
          
          # Environmental and Social Impact
          "web3 environmental initiatives",
          "blockchain environmental solutions",
          "climate DAOs",
          "carbon credit tokens",
          "environmental impact tracking",
          "sustainable blockchain scaling",
          "green consensus mechanisms",
          "solarpunk technology",
          
          # Community and Social
          "community driven development",
          "impact evaluations",
          "community coordination tools",
          "social token design",
          "community ownership",
          "stakeholder capitalism",
          "public goods",
          "coordination failures",
          "web3 social impact",
          
          # Research and Implementation
          "governance research",
          "token engineering research",
          "impact measurement",
          "coordination experiments",
          "mechanism design research",
          
      ]
      self.logger.info(f"Initialized relevance checker with {len(self.relevant_topics)} topics")

    def check_relevance(self, mention_text: str) -> bool:
        """Check if mention is relevant to GreenPill topics and missions"""
        try:
            # Convert mention to lowercase for comparison
            mention_lower = mention_text.lower()
            
            # Check for direct matches first (optimization)
            direct_matches = any(topic.lower() in mention_lower for topic in self.relevant_topics)
            if direct_matches:
                self.logger.info(f"Direct topic match found in: {mention_text[:50]}...")
                return True
                
            # Check for basic questions about topics
            question_starts = ["what is", "what's", "whats", "what are", "how does", "why is", "can you explain", "tell me about"]
            is_question = any(mention_lower.startswith(q) for q in question_starts)
            
            # Create prompt with context for GPT
            topics_summary = ", ".join(sorted(set(self.relevant_topics)))
            
            prompt = f"""Determine if this tweet/question is relevant to GreenPill's missions, blockchain environmental solutions, and web3 social impact. 
    
            Consider it relevant if:
            1. It asks about or mentions any of these topics: {topics_summary}
            2. It's asking about core GreenPill concepts (even in simple terms like "what is X?")
            3. It's related to greenpill.network, greenpill books, or regenerative crypto concepts
            4. It's discussing blockchain environmental solutions or web3 social impact
            5. {"It's asking a basic question about any of these topics" if is_question else ""}
    
            Tweet: "{mention_text}"
            
            Return only 'relevant' or 'not relevant'."""
    
            response = self.openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )
            
            is_relevant = response.choices[0].message.content.strip().lower() == 'relevant'
            self.logger.info(f"Relevance check for '{mention_text[:50]}...': {is_relevant}")
            return is_relevant
            
        except Exception as e:
            self.logger.error(f"Error checking relevance: {e}")
            return False

    def test_reply(self, mention_text: str) -> Optional[TestResult]:
      """Enhanced reply testing with result tracking"""
      self.console.print("\n=== Testing Bot Reply ===", style="bold green")
      self.console.print(f"Mention: {mention_text}", style="blue")
      
      try:
          is_relevant = self.check_relevance(mention_text)
          self.console.print(f"\nRelevance Check: {'Relevant' if is_relevant else 'Not Relevant'}", 
                           style="green" if is_relevant else "red")
          
          result = TestResult(
              timestamp=datetime.now().isoformat(),
              type="reply",
              content=mention_text,
              char_count=len(mention_text),
              is_relevant=is_relevant
          )
          
          if is_relevant:
              try:
                  # Use the base class's generate_reply method
                  reply = self.generate_reply(mention_text)
                  self.console.print(f"\nBot Reply: {reply}", style="cyan")
                  self.console.print(f"Character count: {len(reply)}", style="yellow")
                  result.response = reply
              except Exception as e:
                  error_msg = f"Error generating reply: {str(e)}"
                  self.console.print(error_msg, style="bold red")
                  result.error = error_msg
          else:
              self.console.print("\nSkipping reply - mention not relevant to bot's purpose", 
                               style="yellow")
          
          self.test_results.append(result)
          return result
          
      except Exception as e:
          self.logger.error(f"Error in test_reply: {e}")
          return None

    def save_test_results(self, filename: str = "test_results.json"):
        """Save test results to file"""
        try:
            with open(filename, 'w') as f:
                json.dump([asdict(r) for r in self.test_results], f, indent=2)
            self.console.print(f"\nTest results saved to {filename}", style="green")
        except Exception as e:
            self.logger.error(f"Error saving test results: {e}")

    def display_test_statistics(self):
        """Display comprehensive test statistics"""
        table = Table(title="Test Results Statistics")
        
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        # Calculate statistics
        total_tests = len(self.test_results)
        tweets = [r for r in self.test_results if r.type == "tweet"]
        replies = [r for r in self.test_results if r.type == "reply"]
        relevant_mentions = [r for r in replies if r.is_relevant]
        errors = [r for r in self.test_results if r.error]
        
        # Add rows
        table.add_row("Total Tests Run", str(total_tests))
        table.add_row("Tweets Generated", str(len(tweets)))
        table.add_row("Replies Tested", str(len(replies)))
        table.add_row("Relevant Mentions", str(len(relevant_mentions)))
        table.add_row("Errors Encountered", str(len(errors)))
        
        if tweets:
            avg_tweet_length = sum(t.char_count for t in tweets) / len(tweets)
            table.add_row("Average Tweet Length", f"{avg_tweet_length:.1f}")
        
        self.console.print(table)

def run_interactive_test():
    """Enhanced interactive testing interface"""
    bot = GreenPillBotTester()
    console = Console()
    
    console.print("\n🤖 GreenPillBot Testing Interface", style="bold green")
    console.print("\nCommands:", style="yellow")
    console.print("- 'tweet': Generate and test a new tweet")
    console.print("- 'reply': Test reply to a mention")
    console.print("- 'stats': Display test statistics")
    console.print("- 'save': Save test results")
    console.print("- 'quit': Exit the program")
    
    while True:
        try:
            console.print("\n" + "="*50, style="blue")
            command = console.input("\nEnter command (tweet/reply/stats/save/quit): ").lower()
            
            if command == 'quit':
                if bot.test_results:
                    save = console.input("\nSave test results before quitting? (y/n): ").lower()
                    if save == 'y':
                        bot.save_test_results()
                console.print("\nGoodbye! 👋", style="green")
                break
                
            elif command == 'tweet':
                try:
                    tweet = bot.generate_tweet()
                    console.print(f"\nGenerated Tweet: {tweet}", style="cyan")
                    console.print(f"Character count: {len(tweet)}", style="yellow")
                    
                    result = TestResult(
                        timestamp=datetime.now().isoformat(),
                        type="tweet",
                        content=tweet,
                        char_count=len(tweet)
                    )
                    bot.test_results.append(result)
                    
                except Exception as e:
                    console.print(f"Error generating tweet: {e}", style="bold red")
                    
            elif command == 'reply':
                mention = console.input("Enter a test mention: ")
                bot.test_reply(mention)
                
            elif command == 'stats':
                bot.display_test_statistics()
                
            elif command == 'save':
                filename = console.input("Enter filename (default: test_results.json): ")
                if not filename:
                    filename = "test_results.json"
                bot.save_test_results(filename)
                
            else:
                console.print("Invalid command. Use 'tweet', 'reply', 'stats', 'save', or 'quit'", 
                            style="red")
                
        except KeyboardInterrupt:
            console.print("\nOperation cancelled by user", style="yellow")
        except Exception as e:
            console.print(f"\nUnexpected error: {e}", style="bold red")
            logging.error(f"Unexpected error in interactive test: {e}")

if __name__ == "__main__":
    run_interactive_test()