from simulator import GreenPillBotTester
import json
from datetime import datetime
from rich.console import Console
from dataclasses import dataclass, asdict


# this file is meant to finetune tweets by selecting desirable patten mnually

@dataclass
class Tweet:
    content: str
    created_at: str
    posted: bool = False

class TweetGenerator:
    def __init__(self):
        self.bot = GreenPillBotTester()
        self.console = Console()
        self.tweets_file = "approved_tweets.json"
        
    def generate_tweets(self, count: int):
        approved_tweets = []
        
        self.console.print(f"\nGenerating {count} tweets for review...", style="blue")
        
        for i in range(count):
            try:
                tweet = self.bot.generate_tweet()
                self.console.print(f"\n[{i+1}/{count}] Generated tweet:", style="cyan")
                self.console.print(tweet)
                self.console.print(f"Characters: {len(tweet)}", style="yellow")
                
                if self.console.input("\nApprove this tweet? (y/n): ").lower() == 'y':
                    tweet_record = Tweet(
                        content=tweet,
                        created_at=datetime.now().isoformat()
                    )
                    approved_tweets.append(tweet_record)
                    self.console.print("Tweet approved!", style="green")
                    
            except Exception as e:
                self.console.print(f"Error generating tweet: {e}", style="bold red")
                
        return approved_tweets
    
    def save_tweets(self, tweets):
        try:
            # Load existing tweets first
            existing_tweets = []
            try:
                with open(self.tweets_file, 'r') as f:
                    existing_tweets = [Tweet(**t) for t in json.load(f)]
            except FileNotFoundError:
                pass
            
            # Add new tweets
            all_tweets = existing_tweets + tweets
            
            # Save all tweets
            with open(self.tweets_file, 'w') as f:
                json.dump([asdict(t) for t in all_tweets], f, indent=2)
            self.console.print(f"\nSaved {len(tweets)} new tweets to {self.tweets_file}", style="green")
            
        except Exception as e:
            self.console.print(f"Error saving tweets: {e}", style="bold red")

def main():
    generator = TweetGenerator()
    console = Console()
    
    console.print("\n🤖 Tweet Generator Interface", style="bold green")
    
    while True:
        try:
            count = int(console.input("\nHow many tweets to generate? (or 0 to quit): "))
            
            if count == 0:
                break
                
            new_tweets = generator.generate_tweets(count)
            
            if new_tweets:
                if console.input("\nSave approved tweets? (y/n): ").lower() == 'y':
                    generator.save_tweets(new_tweets)
            
        except KeyboardInterrupt:
            console.print("\nOperation cancelled", style="yellow")
            break
        except ValueError:
            console.print("Please enter a valid number", style="red")
        except Exception as e:
            console.print(f"Error: {e}", style="bold red")
    
    console.print("\nGoodbye! 👋", style="green")

if __name__ == "__main__":
    main()