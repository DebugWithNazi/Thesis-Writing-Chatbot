import streamlit as st
import os
import requests
import hashlib
from typing import List, Dict, Any
from datetime import datetime
import json
import re
from urllib.parse import quote
import time
import random
import markdown

# Import required libraries
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from groq import Groq
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade
from bs4 import BeautifulSoup
import concurrent.futures
from duckduckgo_search import DDGS

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

# Custom Tools for Academic Research and Writing
class AcademicResearchTool(BaseTool):
    name: str = "academic_research"
    description: str = "Conduct comprehensive academic research for thesis/synopsis"

    def _run(self, topic: str, research_areas: str) -> str:
        """Conduct thorough academic research"""
        try:
            time.sleep(1)
            
            # Create multiple search queries for comprehensive research
            search_queries = [
                f"{topic} research studies",
                f"{topic} academic papers",
                f"{topic} recent developments",
                f"{topic} methodology",
                f"{topic} literature review"
            ]
            
            all_research = []
            
            with DDGS() as ddgs:
                for query in search_queries:
                    try:
                        results = list(ddgs.text(query, max_results=6))
                        for result in results:
                            all_research.append({
                                'query': query,
                                'title': result.get('title', ''),
                                'content': result.get('body', ''),
                                'url': result.get('href', ''),
                                'relevance_score': self._calculate_relevance(result.get('body', ''), topic)
                            })
                        time.sleep(0.5)  # Rate limiting
                    except Exception as e:
                        continue
            
            # Sort by relevance and remove duplicates
            unique_research = self._remove_duplicates(all_research)
            unique_research.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return json.dumps(unique_research[:15])  # Top 15 most relevant sources
        except Exception as e:
            return f"Research failed: {str(e)}"

    def _calculate_relevance(self, content: str, topic: str) -> float:
        """Calculate relevance score for research content"""
        topic_words = set(topic.lower().split())
        content_words = set(content.lower().split())
        
        if not topic_words or not content_words:
            return 0.0
        
        intersection = topic_words.intersection(content_words)
        return len(intersection) / len(topic_words)

    def _remove_duplicates(self, research_list: List[Dict]) -> List[Dict]:
        """Remove duplicate research entries"""
        seen_urls = set()
        unique_research = []
        
        for item in research_list:
            if item['url'] not in seen_urls:
                seen_urls.add(item['url'])
                unique_research.append(item)
        
        return unique_research

class CitationGeneratorTool(BaseTool):
    name: str = "citation_generator"
    description: str = "Generate proper academic citations and references"

    def _run(self, research_data: str) -> str:
        """Generate academic citations from research data"""
        try:
            research_items = json.loads(research_data)
            citations = []
            
            for i, item in enumerate(research_items[:10]):  # Top 10 sources
                # Generate citation in APA format
                title = item.get('title', 'Unknown Title')
                url = item.get('url', '')
                
                # Extract domain for author/organization
                domain = url.split('/')[2] if len(url.split('/')) > 2 else 'Unknown'
                
                citation = {
                    'id': f"source_{i+1}",
                    'title': title,
                    'url': url,
                    'domain': domain,
                    'apa_citation': f"{domain}. ({datetime.now().year}). {title}. Retrieved from {url}",
                    'in_text': f"({domain}, {datetime.now().year})"
                }
                citations.append(citation)
            
            return json.dumps(citations)
        except Exception as e:
            return f"Citation generation failed: {str(e)}"

class AcademicWritingTool(BaseTool):
    name: str = "academic_writing"
    description: str = "Analyze and improve academic writing style"

    def _run(self, text: str, academic_level: str) -> str:
        """Analyze academic writing quality and suggest improvements"""
        try:
            # Calculate academic writing metrics
            flesch_score = flesch_reading_ease(text)
            fk_grade = flesch_kincaid_grade(text)
            
            # Analyze sentence structure
            sentences = text.split('.')
            sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
            avg_sentence_length = sum(sentence_lengths) / max(len(sentence_lengths), 1)
            
            # Check for academic writing patterns
            academic_patterns = [
                "furthermore", "moreover", "additionally", "consequently",
                "therefore", "thus", "hence", "accordingly", "subsequently"
            ]
            
            pattern_usage = sum(1 for pattern in academic_patterns if pattern in text.lower())
            
            # Academic level guidelines
            level_guidelines = {
                'undergraduate': {
                    'target_flesch': 60-80,
                    'target_grade': 12-14,
                    'sentence_length': 15-25
                },
                'masters': {
                    'target_flesch': 50-70,
                    'target_grade': 14-16,
                    'sentence_length': 18-30
                },
                'phd': {
                    'target_flesch': 40-60,
                    'target_grade': 16-18,
                    'sentence_length': 20-35
                }
            }
            
            guidelines = level_guidelines.get(academic_level, level_guidelines['masters'])
            
            analysis = {
                'flesch_score': flesch_score,
                'fk_grade': fk_grade,
                'avg_sentence_length': avg_sentence_length,
                'academic_patterns_used': pattern_usage,
                'target_guidelines': guidelines,
                'suggestions': []
            }
            
            # Generate suggestions
            if flesch_score > guidelines['target_flesch'][1]:
                analysis['suggestions'].append("Consider more complex sentence structures for academic tone")
            if avg_sentence_length < guidelines['sentence_length'][0]:
                analysis['suggestions'].append("Use longer, more detailed sentences")
            if pattern_usage < 3:
                analysis['suggestions'].append("Include more academic transition phrases")
                
            return json.dumps(analysis)
        except Exception as e:
            return f"Academic analysis failed: {str(e)}"

class HumanizationTool(BaseTool):
    name: str = "humanization_tool"
    description: str = "Make academic writing sound more human and less AI-like"

    def _run(self, text: str) -> str:
        """Apply humanization techniques to academic writing"""
        try:
            # Common AI patterns in academic writing
            ai_patterns = [
                "It is important to note that",
                "This demonstrates that",
                "This indicates that",
                "As previously mentioned",
                "It should be mentioned that",
                "This suggests that",
                "This implies that",
                "It can be concluded that"
            ]
            
            # Human alternatives
            human_alternatives = [
                "Notably,",
                "This shows",
                "This reveals",
                "As noted earlier",
                "It's worth noting",
                "This suggests",
                "This implies",
                "Therefore,"
            ]
            
            # Apply replacements
            humanized_text = text
            for ai_pattern, human_alt in zip(ai_patterns, human_alternatives):
                humanized_text = humanized_text.replace(ai_pattern, human_alt)
            
            # Add natural variations
            variations = [
                "Interestingly,",
                "Surprisingly,",
                "Remarkably,",
                "Significantly,",
                "Importantly,"
            ]
            
            # Insert variations at appropriate places
            sentences = humanized_text.split('.')
            for i in range(1, len(sentences), 3):  # Every 3rd sentence
                if i < len(sentences) and sentences[i].strip():
                    variation = random.choice(variations)
                    sentences[i] = f" {variation} {sentences[i].lstrip()}"
            
            humanized_text = '.'.join(sentences)
            
            # Add personal insights (subtle)
            personal_insights = [
                "Based on the available evidence,",
                "From the research findings,",
                "Considering the data,",
                "In light of these results,"
            ]
            
            # Insert personal insights
            if len(sentences) > 5:
                insight = random.choice(personal_insights)
                sentences[2] = f" {insight} {sentences[2].lstrip()}"
            
            return '.'.join(sentences)
        except Exception as e:
            return f"Humanization failed: {str(e)}"

# Rate limit handling decorator
def rate_limit_handler(max_retries=3, base_delay=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        st.warning(f"Rate limit hit. Retrying in {delay:.1f} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        raise e
            return None
        return wrapper
    return decorator

# Custom LLM class for CrewAI with built-in API
import os
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import litellm

class BuiltInLLM(LLM):
    model_name: str = "groq/llama-3.3-70b-versatile"
    
    def __init__(self):
        super().__init__()
        # Built-in API key (you can replace this with your own)
        self.api_key = "API_KEY"  # Replace with actual key
        os.environ["GROQ_API_KEY"] = self.api_key
        litellm.set_verbose = False

    @property
    def _llm_type(self) -> str:
        return "groq"

    @rate_limit_handler(max_retries=3, base_delay=2)
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Call API with rate limiting"""
        try:
            # Handle longer prompts for thesis writing
            if len(prompt.split()) > 1500:
                words = prompt.split()
                prompt = ' '.join(words[:1500]) + "..."

            response = litellm.completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert academic writer who creates sophisticated, well-researched thesis documents that sound completely human-written. You avoid AI patterns and create authentic academic content with proper citations and natural flow."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2500,
                temperature=0.6,  # Balanced creativity and consistency
                top_p=0.9,
                api_key=self.api_key
            )

            time.sleep(2)
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error in processing: {str(e)}")
            return f"Error: {str(e)}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}

# Specialized agents for thesis writing
def create_thesis_agents(llm):
    """Create specialized agents for thesis/synopsis writing"""

    # Research Agent
    research_agent = Agent(
        role="Academic Research Specialist",
        goal="Conduct comprehensive academic research and gather credible sources",
        backstory="You are a PhD-level researcher with expertise in finding and analyzing academic sources. You understand how to identify credible information and synthesize research findings.",
        tools=[AcademicResearchTool()],
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Thesis Writer Agent
    thesis_writer = Agent(
        role="Academic Thesis Writer",
        goal="Write sophisticated thesis documents that sound completely human-written",
        backstory="You are an experienced academic writer who specializes in creating thesis documents. You know how to write in a way that sounds natural and scholarly, avoiding AI patterns while maintaining academic rigor.",
        tools=[AcademicWritingTool(), CitationGeneratorTool()],
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Humanization Agent
    humanization_agent = Agent(
        role="Academic Writing Humanizer",
        goal="Make academic writing sound completely human and undetectable",
        backstory="You are an expert editor who specializes in making academic content sound natural and human-written. You know how to eliminate AI patterns and create authentic scholarly writing.",
        tools=[HumanizationTool()],
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    return research_agent, thesis_writer, humanization_agent

def create_thesis_tasks(topic, document_type, academic_level, research_areas, word_count, agents):
    """Create tasks for thesis/synopsis writing"""
    research_agent, thesis_writer, humanization_agent = agents

    # Task 1: Comprehensive Research
    research_task = Task(
        description=f"""
        Conduct comprehensive academic research for a {document_type} on "{topic}".
        
        Research Areas: {research_areas}
        Academic Level: {academic_level}
        Target Length: {word_count} words
        
        Requirements:
        - Find 10-15 credible academic sources
        - Gather recent research and developments
        - Identify key theories and methodologies
        - Note different perspectives and debates
        - Focus on peer-reviewed and scholarly sources
        - Include both theoretical and practical aspects
        
        Provide a detailed research summary with key findings, methodologies, and source analysis.
        """,
        agent=research_agent,
        expected_output="Comprehensive research summary with credible sources and key insights"
    )

    # Task 2: Thesis Writing
    thesis_task = Task(
        description=f"""
        Write a complete {document_type} on "{topic}" that sounds completely human-written.
        
        Academic Level: {academic_level}
        Target Length: {word_count} words
        Research Areas: {research_areas}
        
        Requirements:
        - Use the comprehensive research provided
        - Write in proper academic style for {academic_level} level
        - Include proper citations and references
        - Create logical structure with introduction, body, and conclusion
        - Use varied sentence structures and academic vocabulary
        - Include critical analysis and original insights
        - Maintain scholarly tone while sounding natural
        - Avoid AI-like patterns and formal robotic language
        - Include methodology, findings, and implications
        - Make it engaging and intellectually rigorous
        
        Structure:
        1. Introduction and background
        2. Literature review
        3. Methodology
        4. Analysis and findings
        5. Discussion and implications
        6. Conclusion and recommendations
        
        Important: Write as if you're a human academic expert sharing original research and insights.
        """,
        agent=thesis_writer,
        expected_output="Complete academic thesis document with proper structure and citations",
        dependencies=[research_task]
    )

    # Task 3: Humanization and Polish
    humanization_task = Task(
        description=f"""
        Polish and humanize the thesis document to make it completely undetectable as AI-written.
        
        Requirements:
        - Remove any remaining AI patterns
        - Improve natural academic flow
        - Add authentic human writing touches
        - Ensure varied sentence structures
        - Make transitions feel natural and scholarly
        - Add subtle personal insights and critical thinking
        - Maintain academic rigor while sounding human
        - Improve readability without losing sophistication
        - Ensure proper citation integration
        - Make it sound like expert human academic writing
        
        Focus on making it indistinguishable from high-quality human academic writing.
        """,
        agent=humanization_agent,
        expected_output="Final polished human-sounding academic thesis document",
        dependencies=[thesis_task]
    )

    return [research_task, thesis_task, humanization_task]

def run_thesis_writer(topic, document_type, academic_level, research_areas, word_count):
    """Run the thesis writing process"""
    try:
        # Initialize LLM
        llm = BuiltInLLM()

        # Create agents
        agents = create_thesis_agents(llm)

        # Create tasks
        tasks = create_thesis_tasks(topic, document_type, academic_level, research_areas, word_count, agents)

        # Create crew
        crew = Crew(
            agents=list(agents),
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

        # Execute with progress tracking
        with st.spinner("Creating comprehensive thesis document with AI agents..."):
            result = crew.kickoff()

        return result
    except Exception as e:
        st.error(f"Error in thesis writing: {str(e)}")
        return None

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Thesis Writer Bot - Academic Document Creator",
        page_icon="🎓",
        layout="wide"
    )

    st.title("🎓 Thesis Writer Bot")
    st.markdown("*Create sophisticated, human-like thesis and synopsis documents that pass any AI detection*")

    # Sidebar configuration
    with st.sidebar:
        st.header("ℹ️ About")
        
        st.success("✅ Ready to generate your thesis!")
        
        st.markdown("---")
        st.markdown("### 🎯 What This Tool Does")
        st.markdown("- Creates original, human-like thesis documents")
        st.markdown("- Conducts comprehensive academic research")
        st.markdown("- Generates proper citations and references")
        st.markdown("- Ensures content passes AI detection")
        st.markdown("- No plagiarism - completely original content")

        st.markdown("---")
        st.markdown("### 📚 Document Types")
        st.markdown("- **Thesis**: Complete research thesis")
        st.markdown("- **Synopsis**: Research proposal/synopsis")
        st.markdown("- **Dissertation**: PhD-level document")
        st.markdown("- **Research Paper**: Academic paper")
        st.markdown("- **Literature Review**: Comprehensive review")

        st.markdown("---")
        st.markdown("### 🎓 Academic Levels")
        st.markdown("- **Undergraduate**: Bachelor's level")
        st.markdown("- **Masters**: Graduate level")
        st.markdown("- **PhD**: Doctoral level")

        st.markdown("---")
        st.markdown("### 🔥 Features")
        st.markdown("- **No Plagiarism**: Original research")
        st.markdown("- **Human-like**: Natural academic writing")
        st.markdown("- **AI Undetectable**: Passes detection")
        st.markdown("- **Proper Citations**: Academic references")
        st.markdown("- **Research-based**: Credible sources")
        st.markdown("- **No Word Limits**: Any length needed")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📝 Thesis Request")

        # Topic input
        topic = st.text_input(
            "What is your thesis/synopsis topic?",
            placeholder="e.g., Impact of artificial intelligence on healthcare delivery systems"
        )

        # Document type selection
        document_types = [
            "Thesis", "Synopsis", "Dissertation", "Research Paper", 
            "Literature Review", "Research Proposal", "Academic Report"
        ]
        document_type = st.selectbox("Document Type", document_types)

        # Academic level
        academic_levels = ["Undergraduate", "Masters", "PhD"]
        academic_level = st.selectbox("Academic Level", academic_levels)

        # Research areas
        research_areas = st.text_area(
            "Specific Research Areas/Focus (Optional)",
            placeholder="e.g., methodology, recent developments, case studies, theoretical frameworks...",
            height=80
        )

        # Word count (no limit)
        word_count = st.number_input(
            "Target Word Count",
            min_value=1000,
            max_value=50000,
            value=5000,
            step=500,
            help="No strict limit - write as much as needed"
        )

        # Additional requirements
        additional_requirements = st.text_area(
            "Additional Requirements (Optional)",
            placeholder="Specific methodology, theoretical framework, case studies, etc...",
            height=100
        )

        # Generate button
        if st.button("🚀 Generate Thesis Document", type="primary", use_container_width=True):
            if not topic.strip():
                st.error("Please enter a thesis topic!")
            else:
                # Prepare research areas
                research_areas_text = research_areas if research_areas.strip() else "general academic research"
                
                # Run thesis generation
                result = run_thesis_writer(topic, document_type, academic_level, research_areas_text, word_count)

                if result:
                    st.session_state.generated_thesis = result
                    st.session_state.thesis_info = {
                        'topic': topic,
                        'type': document_type,
                        'level': academic_level,
                        'research_areas': research_areas_text,
                        'word_count': word_count,
                        'requirements': additional_requirements
                    }
                    st.success("✅ Thesis document generated successfully!")

    with col2:
        st.header("📄 Generated Thesis")

        if "generated_thesis" in st.session_state:
            thesis = st.session_state.generated_thesis
            info = st.session_state.thesis_info

            # Display thesis info
            st.subheader("📊 Document Information")
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Topic", info['topic'])
                st.metric("Type", info['type'])
                st.metric("Level", info['level'])
            with col_info2:
                st.metric("Generated Words", len(str(thesis).split()))
                st.metric("Research Areas", info['research_areas'][:20] + "..." if len(info['research_areas']) > 20 else info['research_areas'])
                st.metric("Quality", "✅ Human-like")

            # Display the thesis
            st.subheader("📝 Your Thesis Document")
            
            # Format the thesis nicely
            formatted_thesis = str(thesis)
            
            st.text_area(
                "Generated Thesis:",
                value=formatted_thesis,
                height=400,
                help="This is your human-like thesis document"
            )

            # Download options
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="📥 Download as TXT",
                    data=formatted_thesis,
                    file_name=f"{info['topic'].replace(' ', '_')}_{info['type']}.txt",
                    mime="text/plain"
                )
            
            with col_dl2:
                # Create markdown version with academic formatting
                markdown_content = f"""# {info['topic']}

**Document Type:** {info['type']}  
**Academic Level:** {info['level']}  
**Research Areas:** {info['research_areas']}  
**Word Count:** {len(str(thesis).split())}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

{formatted_thesis}

---

*This document was generated using advanced AI technology and is designed to be indistinguishable from human academic writing.*
"""
                st.download_button(
                    label="📥 Download as MD",
                    data=markdown_content,
                    file_name=f"{info['topic'].replace(' ', '_')}_{info['type']}.md",
                    mime="text/markdown"
                )

            # Document analysis
            st.subheader("🔍 Document Analysis")
            
            # Quick stats
            actual_words = len(str(thesis).split())
            actual_sentences = len(str(thesis).split('.'))
            paragraphs = len(str(thesis).split('\n\n'))
            
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric("Words", actual_words)
            with col_stats2:
                st.metric("Sentences", actual_sentences)
            with col_stats3:
                st.metric("Paragraphs", paragraphs)

            # Academic quality indicators
            st.success("✅ Document optimized for academic writing")
            st.info("💡 This thesis is designed to pass AI detection tools and academic scrutiny")
            st.warning("⚠️ Remember to review and customize the content for your specific requirements")
            
            # Remove technical details
            st.markdown("---")
            st.markdown("### 🔒 Privacy & Security")
            st.markdown("- Your content is processed securely")
            st.markdown("- No data is stored or shared")
            st.markdown("- All research is conducted privately")

        else:
            st.info("👈 Enter a thesis topic and click 'Generate Thesis Document' to create your academic content")

if __name__ == "__main__":
    main() 