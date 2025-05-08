from llama_parse import LlamaParse
from dotenv import load_dotenv
import os
import re


load_dotenv()

class DocumentParser:
    def __init__(self):
        self.parser = LlamaParse(
                parse_mode="parse_page_with_agent",
                #vendor_multimodal_model_name="openai-gpt4o",
                #vendor_multimodal_api_key=os.environ.get("OPENAI_API_KEY"),

                result_type="markdown",
                system_prompt_append="When a table contains merged cells (cells that span across multiple rows/columns), " \
                "you must expand the table by duplicating the value of the merged cell into every individual cell it spans. " \
                "Do not let page break breaks the table." \
                "The 'Get Help' text is a page header, do not mix it with the main content, must put it in a separate line." \
                "Do not use newline in a table cell as that will break the table. Use <br/> instead.",
            )
    
    def ingest(self, file_name):
        result = self.parse(file_name)

        clean_result = self.clean(result)

        return clean_result

    async def parse(self, file_name):
        result = await self.parser.aparse(file_name)
        content = ""

        for page in result.pages:
            content += page.md
            
        return content
    
    def clean(self, text: str) -> str:
        # Split the document by lines
        lines = text.split('\n')
        cleaned_lines = []
        
        # Flag to track if we're in the footer section
        in_footer = False
        
        for line in lines:
            # Clean up "Get Help" header
            if re.match(r"^.*[#\*]*\s*Get\s+Help\s*$", line):
                line = re.sub(r"[#\*]*\s*Get\s+Help\s*$", "", line)
                
            # Check if we've reached the footer (starts with "About CelcomDigi")
            if re.match(r"^[#\*]*\s*About CelcomDigi\s*[#\*]*\s*$", line):
                in_footer = True
                continue
                
            # Skip all lines once we're in the footer section
            if in_footer:
                continue
                
            # Add valid content lines
            cleaned_lines.append(line)
        
            # Join the cleaned lines back into a single text
            cleaned_text = '\n'.join(cleaned_lines)

        return cleaned_text







