from typing import List, Dict, Optional
import random
import logging
from dataclasses import dataclass

@dataclass
class TweetStyle:
    name: str
    template: str
    question_template: Optional[str] = None
    cta_template: Optional[str] = None

class ContentGenerator:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.recent_styles = []
        self.recent_reply_styles = []
        self.max_history = 3
        self._setup_tweet_styles()
        self._setup_reply_styles()

    def _setup_tweet_styles(self):
        """Initialize comprehensive tweet style templates"""
        self.tweet_styles = [
            TweetStyle(
                name="Sarcastic Technical",
                template="lol watching tradfi try to solve coordination problems with spreadsheets while we're out here with pure mathematical proofs of better systems. literally can't make this up 🤣",
                question_template="anyone else notice how tradfi keeps reinventing worse versions of our solutions? what coordination mechanisms are you using that actually work? 🤔"
            ),
            TweetStyle(
                name="Playful Observation",
                template="my code commits are aligned with planetary boundaries while your github is still running on pure hopium. time to level up that impact coefficient fam 💫",
                question_template="checking these impact metrics and wondering - what's your favorite way to measure positive externalities in your projects? 📊"
            ),
            TweetStyle(
                name="Technical Flex",
                template="optimizing governance parameters at 3am because sleep is for those who haven't discovered mechanism design. pure sigma grindset but make it regenerative 😤",
                question_template="deep in these governance simulations. what's your preferred voting mechanism for complex decisions? quadratic looking kinda based rn 🤔"
            ),
            TweetStyle(
                name="Hot Take",
                template="hot take: your ponzinomics degree won't save you from coordination failures. been reviewing impact certs all morning and the math is literally screaming at us 🔥",
                question_template="hot take: most projects sleep on proper mechanism design. what incentive structures are actually working in your DAOs? drop your metrics 📈"
            ),
            TweetStyle(
                name="Chad Energy",
                template="broke: endless meetings about alignment. woke: deploying smart contracts that mathematically guarantee aligned incentives. pure builder energy today 🛠️",
                question_template="broke: trust me bro tokenomics. woke: mathematical proofs of alignment. what's your approach to verifiable impact? show your work fam 🧮"
            ),
            TweetStyle(
                name="Philosophical",
                template="thinking about how we're literally coding better social systems while others debate if web3 has value. the math doesn't care about your opinion ser 🧠",
                question_template="pondering these coordination mechanisms and their long-term effects. what governance experiments are you running that actually scale? 🌱"
            ),
            TweetStyle(
                name="Building Update",
                template="deploying regenerative mechanisms while the market obsesses over zero-sum games. literally can't stop thinking about these impact curves 📈",
                question_template="reviewing our governance participation data - fascinating patterns emerging. what's your secret to maintaining high voter engagement? 🎯"
            ),
            TweetStyle(
                name="Ecosystem Commentary",
                template="watching projects speed run every governance mistake we documented while claiming innovation. ser, the research papers were right there 😅",
                question_template="seeing teams struggle with basic coordination. what's your go-to solution for aligning incentives in practice? wrong answers only 😏"
            ),
            TweetStyle(
                name="Data Driven",
                template="plotting impact metrics vs token distributions and the correlation is literally mathematical poetry. pure signal in a market full of noise 📊",
                question_template="analyzing our impact data and finding some wild patterns. what metrics do you track to measure real coordination success? 📈"
            ),
            TweetStyle(
                name="Community Focus",
                template="vibing with builders who understand that community > capital. your token price is temporary but impact is forever 🌱",
                question_template="fascinated by different community engagement models. what's working best for sustained participation in your DAO? drop your secrets 🤝"
            ),
            TweetStyle(
                name="Protocol Insight",
                template="studying {protocol} source - their coordinator contract uses an elegant state machine pattern for slashing. ~40% gas savings vs naive implementation",
                question_template="reviewing {topic} implementations - what patterns give you the best gas optimization without sacrificing security?"
            ),
            
            TweetStyle(
                name="Research Deep Dive",
                template="page 17 of the {paper} paper completely changes our assumptions about token weighted voting. the math on coordinated stake patterns is wild",
                question_template="finding counterintuitive results in recent {topic} research. what papers changed your understanding of mechanism design?"
            ),
            
            TweetStyle(
                name="Debug Log",
                template="tracked down that governance bug - turns out naive timestamp checks break under certain validator reorg conditions. back to basics",
                question_template="spent 6 hours debugging a subtle {topic} edge case. what's your weirdest coordination failure story?"
            ),
            
            TweetStyle(
                name="Architecture Notes",
                template="rejected another overcomplicated proposal. you can model 90% of coordination games with simple state machines + verifiable exits",
                question_template="reviewing {topic} architecture patterns. what unnecessary complexity have you eliminated from your systems?"
            ),
            
            # Impact & Analytical Variations
            TweetStyle(
                name="Impact Analysis",
                template="ran the numbers: our latest coordination mechanism distributed 47% more rewards to active contributors vs passive holders. incentives matter",
                question_template="analyzing {topic} contribution data. what metrics actually capture value creation in your system?"
            ),
            
            TweetStyle(
                name="System Behavior",
                template="fascinating emergent behavior in testnet: smaller holders naturally form voting blocks to counter whale dynamics. coordination finds a way",
                question_template="observing unexpected {topic} behaviors in prod. what emergent patterns surprised you?"
            ),
            
            # Critical & Thoughtful Variations
            TweetStyle(
                name="Design Critique",
                template="unpopular opinion: your fancy quadratic bonding curve won't fix misaligned incentives. first principles or it won't scale",
                question_template="critiquing common {topic} assumptions. which 'best practices' actually make coordination harder?"
            ),
            
            TweetStyle(
                name="Whiteboard Thoughts",
                template="sketched out our coordination failure modes - 80% trace back to poor state management. complexity is the enemy of correctness",
                question_template="mapping {topic} failure scenarios. where do your systems break under byzantine conditions?"
            ),
            
            # Engineering Focus Variations
            TweetStyle(
                name="Code Review",
                template="reviewing governance PRs. someone snuck in a beautifully minimal coordination mechanism using just events + a merkle root. based",
                question_template="found an elegant {topic} pattern in the codebase. what's your clevarest coordination solution?"
            ),
            
            TweetStyle(
                name="Performance Log",
                template="benchmarking coordination costs: new merkle-based reputation tracking cuts gas by 65% vs on-chain storage. data structures matter",
                question_template="optimizing {topic} operations. what's your best gas saving trick that doesn't sacrifice security?"
            ),
            
        ]

    def _setup_reply_styles(self):
        """Initialize comprehensive reply style templates"""
        self.reply_styles = [
            TweetStyle(
                name="Technical Helper",
                template="seeing you work through these coordination problems is based. been there - try adding quadratic funding to your stack. literally changed our whole game 🎯"
            ),
            TweetStyle(
                name="Playful Guide",
                template="your intuition about mechanism design is spot on fam. i ran these simulations last week and the math is screaming 'yes'. time to level up that governance 📈"
            ),
            TweetStyle(
                name="Knowledge Drop",
                template="ah, the classic governance paradox. threw my best solidity at this last month. peep the greenpill papers - pure mathematical proof of better ways 🧠"
            ),
            TweetStyle(
                name="Builder Energy",
                template="love where your head's at. been coding similar systems all week - the coordination gains are literally off the charts. your instincts are right 🛠️"
            ),
            TweetStyle(
                name="Thought Partner",
                template="fascinating approach to impact certs you're exploring. reminds me of some wild experiments we ran - the data suggests you're onto something big 💫"
            ),
            TweetStyle(
                name="Ecosystem Wisdom",
                template="seen many projects try this path - the ones that succeeded all had one thing in common: based mechanism design. happy to share our research 📚"
            ),
            TweetStyle(
                name="Math Enjoyer",
                template="getting excited about your token model. ran some quick numbers and the math is surprisingly elegant. kind of based ngl 🔢"
            ),
            TweetStyle(
                name="Future Vision",
                template="while others debate web2 solutions, you're already modeling web3 coordination. this kind of thinking literally builds new worlds 🌍"
            ),
            TweetStyle(
                name="Community Champion",
                template="your focus on community-driven development is exactly what we need. been seeing similar patterns in our impact data. keep building the future 🌱"
            ),
            TweetStyle(
                name="Research Buddy",
                template="fascinating research direction. our simulations showed similar results - have you considered adding conviction voting to the mix? could be game-changing 🔬"
            ),
            
            TweetStyle(
                name="Code Reviewer",
                template="interesting approach. we tried similar in prod - key insight was separating coordination logic from state management. happy to share the patterns"
            ),
            
            TweetStyle(
                name="Data Analyst",
                template="solid intuition. our metrics show similar patterns - coordination efficiency peaks when governance tokens align with usage rights. DM for details"
            ),
            
            TweetStyle(
                name="Systems Architect",
                template="been solving this exact problem - turns out simple state machines + verifiable exits handle 90% of coordination cases. check the latest commit"
            ),
            
            TweetStyle(
                name="Security Mindset",
                template="careful with that pattern - under certain conditions it can lead to coordination failures. found a cleaner approach using merkle proofs"
            ),
            
            TweetStyle(
                name="Infrastructure Dev",
                template="tracked similar issues in our system. solution was moving coordination logic off-chain with robust verification. massive gas savings"
            ),
            
            TweetStyle(
                name="Research Engineer",
                template="fascinating direction. latest papers show promising results combining this with conviction voting. happy to share references"
            )
                    ]

    def get_tweet_style(self, include_questions: bool = False) -> TweetStyle:
        """Get a tweet style while maintaining diversity"""
        try:
            available_styles = [s for s in self.tweet_styles if s not in self.recent_styles]
            
            if not available_styles:
                self.recent_styles = self.recent_styles[-1:]
                available_styles = [s for s in self.tweet_styles if s not in self.recent_styles]
            
            # Filter styles based on whether they support questions if needed
            if include_questions:
                available_styles = [s for s in available_styles if s.question_template]
            
            style = random.choice(available_styles)
            self.recent_styles.append(style)
            if len(self.recent_styles) > self.max_history:
                self.recent_styles.pop(0)
            
            return style
        except Exception as e:
            self.logger.error(f"Error getting tweet style: {e}")
            # Fall back to first style if error occurs
            return self.tweet_styles[0]
    
    def _get_technical_detail(self) -> str:
       """Get a random technical detail to add diversity"""
       details = [
           "gas savings",
           "throughput increase",
           "better distribution",
           "coordination efficiency",
           "3x faster finality",
           "reduction in failures",
           "lower variance",
           "improvement in...",
           "confidence interval",
           "optimization in...",
       ]
       return random.choice(details)

    def get_reply_style(self) -> TweetStyle:
        """Get a reply style while maintaining diversity"""
        try:
            available_styles = [s for s in self.reply_styles if s not in self.recent_reply_styles]
            
            if not available_styles:
                self.recent_reply_styles = self.recent_reply_styles[-1:]
                available_styles = [s for s in self.reply_styles if s not in self.recent_reply_styles]
            
            style = random.choice(available_styles)
            self.recent_reply_styles.append(style)
            if len(self.recent_reply_styles) > self.max_history:
                self.recent_reply_styles.pop(0)
            
            return style
        except Exception as e:
            self.logger.error(f"Error getting reply style: {e}")
            return self.reply_styles[0]

    def format_tweet_prompt(self, content: str, style: TweetStyle, include_question: bool = False) -> str:
        """Format prompt for tweet generation"""
        template = style.question_template if include_question and style.question_template else style.template
        
        prompt = f"""Generate a tweet based on this content: "{content}"

        Style: {style.name}
        Example: {template}

        General guidelines:
        - Be technical but witty
        - Mix deep insights with humor
        - Use "I" statements naturally
        - Can be edgy but not offensive
        - Reference technical concepts playfully
        - Occasionally drop a "based", "literally", "pure" naturally
        - Sound like a real builder having fun
        - Add emojis sparingly but effectively
        - {'Include a specific question about implementation details or metrics' if include_question else ''}

        Keep tweet strictly under 200 characters. Be creative and diverse - avoid starting with "Just" or common patterns."""
        
        return prompt

    def format_reply_prompt(self, mention_text: str, context: str, style: TweetStyle) -> str:
        """Format prompt for reply generation"""
        return f"""Generate a reply to this tweet: "{mention_text}"
        
        Using this context when relevant: "{context}"

        Style: {style.name}
        Example: {style.template}

        General guidelines:
        - Be helpful and encouraging while staying witty
        - Mix technical insights with playful language
        - Use "I" statements and personal experiences
        - Can be edgy but always supportive
        - Reference GreenPill concepts naturally
        - Drop technical terms with context
        - Sound like a builder helping another builder
        - Add emojis thoughtfully
        - Make it feel like a real conversation
        - Keep guidance practical and actionable

        Keep reply strictly under 200 characters. Be creative and diverse - make each reply feel unique and personal."""

    def get_style_stats(self) -> Dict:
        """Get statistics about style usage"""
        return {
            'total_tweet_styles': len(self.tweet_styles),
            'total_reply_styles': len(self.reply_styles),
            'recent_tweet_styles': [s.name for s in self.recent_styles],
            'recent_reply_styles': [s.name for s in self.recent_reply_styles],
            'question_capable_styles': len([s for s in self.tweet_styles if s.question_template])
        }

if __name__ == "__main__":
    # Test the content generator
    generator = ContentGenerator()
    print("\nTesting ContentGenerator...")
    
    # Test tweet styles
    style = generator.get_tweet_style(include_questions=True)
    print(f"\nSelected Tweet Style: {style.name}")
    print(f"Template: {style.template}")
    if style.question_template:
        print(f"Question Template: {style.question_template}")
    
    # Test reply styles
    reply_style = generator.get_reply_style()
    print(f"\nSelected Reply Style: {reply_style.name}")
    print(f"Template: {reply_style.template}")
    
    # Print stats
    print("\nStyle Statistics:")
    print(generator.get_style_stats())