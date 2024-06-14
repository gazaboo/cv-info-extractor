import ollama
import streamlit as st
from PyPDF2 import PdfReader
from streamlit_pdf_viewer import pdf_viewer

def talk_to_ollama(question, context):
  response = ollama.generate(
     model='llama3', 
     format='json', 
     stream=False,
     prompt='\n\n'.join([context, question])
  )

  return response



def main():
    st.set_page_config(page_title='Extract info from CV', page_icon=':books:')
    st.header('Extract CV Info')

    pdf_doc = st.file_uploader('Import CV here')

    if st.button('Process'):
        pdf_bytes = pdf_doc.read()
        
        col1, col2 = st.columns(2)
        with col1:
            pdf_viewer(pdf_bytes)
        with col2:
         with st.spinner('Processing'):
            cv_text, _ = get_pdf_text(pdf_doc)
            question = """
            Génère un JSON contenant les informations du candidat en suivant le format suivant : 

            {
                'nom': le nom du candidat (type : str),
                'experience_professionnelle' : liste des experiences professionnelles du candidat (type :  List[str] ),
                'langues': liste des langues parlées par le candidat (type :  List[str] ),
                'diplomes': liste des diplômes ou formations du candidat (type :  List[str] )',
                'programmation' : liste des langages de programmation maîtrisés par le candidat (type :  List[str] )
            }

            Ne rajoute aucun champ supplémentaire au JSON. 
            Tu respecteras présisément le format proposé. 
            Tu ne réponds qu'avec le JSON. 
            """

            answer = talk_to_ollama(question, cv_text)
            st.write('Temps de chargement du modèle :', answer['load_duration'] / 10**9, '(s)')
            st.write("Temps d'évaluation du prompt :", answer['prompt_eval_duration'] / 10**9, '(s)')
            st.write("Temps de génération du prompt :", answer['eval_duration'] / 10**9, '(s)')
            st.write('Inférence : ')
            st.json(answer['response'])


def get_pdf_text(pdf_doc):
    text = ''
    meta_info = dict()
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text().replace('\n', ' ')
        meta_info[page.extract_text().replace('\n', ' ')] = pdf_doc.name
    return text, meta_info


if __name__ == '__main__':
    main()
