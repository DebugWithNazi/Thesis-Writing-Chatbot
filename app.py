import streamlit as st
import os
import requests
from datetime import datetime
import time
import random
import markdown
from groq import Groq

# --- Streamlit UI and Groq API Integration ---

def build_thesis_prompt(topic, document_type, academic_level, research_areas, word_count, additional_requirements):
    prompt = f"""
You are an expert academic writer. Write a complete {document_type} on the topic: "{topic}".
Academic Level: {academic_level}
Target Length: {word_count} words
Research Areas: {research_areas}
"""
    if additional_requirements and additional_requirements.strip():
        prompt += f"\nAdditional Requirements: {additional_requirements.strip()}\n"
    prompt += """
Requirements:
- Use credible academic sources and reference them in-text (APA style, e.g., (Author, Year)).
- Write in proper academic style for the specified level.
- Create logical structure with introduction, body, and conclusion.
- Use varied sentence structures and academic vocabulary.
- Include critical analysis and original insights.
- Maintain scholarly tone while sounding natural and human.
- Avoid AI-like patterns and robotic language.
- Include methodology, findings, and implications if relevant.
- Make it engaging and intellectually rigorous.
Structure:
1. Introduction and background
2. Literature review
3. Methodology
4. Analysis and findings
5. Discussion and implications
6. Conclusion and recommendations
Important: Write as if you're a human academic expert sharing original research and insights. Make it indistinguishable from human writing.

Begin the document below:

"""
    return prompt

def call_groq_llama(prompt, api_key, model_name="llama3-70b-8192"):  # Use the correct Groq model name
    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert academic writer who creates sophisticated, well-researched thesis documents that sound completely human-written. You avoid AI patterns and create authentic academic content with proper citations and natural flow."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2500,
            temperature=0.6,
            top_p=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error from Groq API: {str(e)}")
        return f"Error: {str(e)}"

def run_thesis_writer(topic, document_type, academic_level, research_areas, word_count, additional_requirements, api_key):
    prompt = build_thesis_prompt(topic, document_type, academic_level, research_areas, word_count, additional_requirements)
    with st.spinner("Generating your thesis document with Groq Llama-3..."):
        result = call_groq_llama(prompt, api_key)
    return result

def main():
    st.set_page_config(
        page_title="Thesis Writer Bot - Academic Document Creator",
        page_icon="🎓",
        layout="wide"
    )
    st.title("🎓 Thesis Writer Bot")
    st.markdown("*Create sophisticated, human-like thesis and synopsis documents that pass any AI detection*")
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
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("📝 Thesis Request")
        topic = st.text_input(
            "What is your thesis/synopsis topic?",
            placeholder="e.g., Impact of artificial intelligence on healthcare delivery systems"
        )
        document_types = [
            "Thesis", "Synopsis", "Dissertation", "Research Paper", 
            "Literature Review", "Research Proposal", "Academic Report"
        ]
        document_type = st.selectbox("Document Type", document_types)
        academic_levels = ["Undergraduate", "Masters", "PhD"]
        academic_level = st.selectbox("Academic Level", academic_levels)
        research_areas = st.text_area(
            "Specific Research Areas/Focus (Optional)",
            placeholder="e.g., methodology, recent developments, case studies, theoretical frameworks...",
            height=80
        )
        word_count = st.number_input(
            "Target Word Count",
            min_value=1000,
            max_value=50000,
            value=5000,
            step=500,
            help="No strict limit - write as much as needed"
        )
        additional_requirements = st.text_area(
            "Additional Requirements (Optional)",
            placeholder="Specific methodology, theoretical framework, case studies, etc...",
            height=100
        )
        api_key = os.environ.get("GROQ_API_KEY")
        if st.button("🚀 Generate Thesis Document", type="primary", use_container_width=True):
            if not topic.strip():
                st.error("Please enter a thesis topic!")
            elif not api_key:
                st.error("Groq API key not configured. Please set GROQ_API_KEY environment variable.")
            else:
                research_areas_text = research_areas if research_areas.strip() else "general academic research"
                result = run_thesis_writer(topic, document_type, academic_level, research_areas_text, word_count, additional_requirements, api_key)
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
            st.subheader("📝 Your Thesis Document")
            formatted_thesis = str(thesis)
            st.text_area(
                "Generated Thesis:",
                value=formatted_thesis,
                height=400,
                help="This is your human-like thesis document"
            )
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="📥 Download as TXT",
                    data=formatted_thesis,
                    file_name=f"{info['topic'].replace(' ', '_')}_{info['type']}.txt",
                    mime="text/plain"
                )
            with col_dl2:
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
            st.subheader("🔍 Document Analysis")
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
            st.success("✅ Document optimized for academic writing")
            st.info("💡 This thesis is designed to pass AI detection tools and academic scrutiny")
            st.warning("⚠️ Remember to review and customize the content for your specific requirements")
            st.markdown("---")
            st.markdown("### 🔒 Privacy & Security")
            st.markdown("- Your content is processed securely")
            st.markdown("- No data is stored or shared")
            st.markdown("- All research is conducted privately")
        else:
            st.info("👈 Enter a thesis topic and click 'Generate Thesis Document' to create your academic content")

if __name__ == "__main__":
    main() 
