import streamlit as st
import requests
import json
import pandas as pd
import re
import matplotlib.pyplot as plt
import os
import base64

# Konfiguracja strony Streamlit
st.set_page_config(page_title="DeepSeek Data Analyzer", layout="wide")

# Odczytanie pliku z promptem instrukcji
@st.cache_data
def load_system_prompt():
    try:
        with open("instruction/prompt_instruction.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return """Jesteś pomocnym asystentem data science. Pomóż użytkownikowi przeszukiwać i analizować dane.
Zawsze najpierw przemyśl problem, a następnie zwróć kod Python do filtrowania danych.
Używaj zmiennych: filtered, filtered_df lub result do przechowywania przefiltrowanych danych."""

# Funkcja do usuwania tagów <think>
def remove_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# Funkcja do generowania odpowiedzi z modelu Deepseek
def generate_response(prompt, model_name="deepseek-r1:14b", max_tokens=500, temperature=0.7):
    """
    Generuje odpowiedź z modelu Deepseek zainstalowanego przez Ollama.
    """
    try:
        # URL do API Ollama (domyślnie działa na localhost:11434)
        url = "http://localhost:11434/api/generate"
        
        # Przygotowanie danych zapytania
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        # Wysłanie zapytania do API Ollama
        with st.spinner('Czekam na odpowiedź modelu...'):
            response = requests.post(url, json=payload)
        
        # Sprawdzenie odpowiedzi
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "Brak odpowiedzi od modelu")
        else:
            return f"Błąd API: {response.status_code}, {response.text}"
            
    except Exception as e:
        return f"Wystąpił błąd: {str(e)}"

# Funkcja do wykonania wygenerowanego kodu
def execute_generated_code(code, df):
    """
    Wykonuje kod na przekazanym DataFrame i zwraca wynik filtrowania.
    """
    try:
        local_vars = {"df": df}
        exec(code, {}, local_vars)

        # Szukamy potencjalnej zmiennej z wynikiem filtrowania
        for var_name in ["filtered", "filtered_df", "result"]:
            if var_name in local_vars and isinstance(local_vars[var_name], pd.DataFrame):
                return local_vars[var_name]

        st.warning("⚠️ Nie znaleziono zmiennej z przefiltrowanym DataFrame.")
        return None

    except Exception as exec_error:
        st.error(f"❌ Błąd wykonania kodu: {exec_error}")
        return None

# Funkcja do pobierania CSV z uploadowanego pliku
def get_csv_download_link(df, filename="filtered_data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Pobierz dane CSV</a>'
    return href

# Główna funkcja aplikacji
def main():
    st.title("📊 DeepSeek Data Analyzer")
    st.write("Zadaj pytanie, aby przeanalizować dane z pliku CSV.")
    
    # Sidebar z ustawieniami
    with st.sidebar:
        st.header("Ustawienia")
        
        # Upload pliku CSV
        uploaded_file = st.file_uploader("Wgraj plik CSV", type=["csv"])
        
        # Wybór modelu
        model_name = st.selectbox(
            "Wybierz model",
            ["deepseek-r1:14b", "llama3:8b", "gemma:7b"]
        )
        
        # Parametry generowania
        temperature = st.slider("Temperatura", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.number_input("Maksymalna liczba tokenów", 100, 2000, 500, 100)

    # Wczytanie danych
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Wczytano plik z {df.shape[0]} wierszami i {df.shape[1]} kolumnami")
            
            # Wyświetlanie przykładowych danych w sidebarze
            with st.sidebar.expander("Podgląd danych"):
                st.dataframe(df.head(5), use_container_width=True)
                
        except Exception as e:
            st.sidebar.error(f"Błąd podczas wczytywania pliku: {e}")
    else:
        # Próba wczytania domyślnego pliku CSV
        try:
            csv_path = "datasets/amazon_sales_data 2025.csv"
            df = pd.read_csv(csv_path)
            st.sidebar.info(f"Używam domyślnego pliku: {csv_path}")
            
            # Wyświetlanie przykładowych danych w sidebarze
            with st.sidebar.expander("Podgląd danych"):
                st.dataframe(df.head(5), use_container_width=True)
                
        except Exception as e:
            st.sidebar.warning(f"Nie udało się wczytać domyślnego pliku: {e}")
            st.warning("⚠️ Proszę wgrać plik CSV, aby rozpocząć analizę")

    # Pole wejściowe dla zapytania użytkownika
    with st.container():
        user_input = st.text_area(
            "Wpisz swoje zapytanie:",
            placeholder="Np. 'Pokaż mi 10 najlepiej sprzedających się produktów' lub 'Znajdź sprzedaż w kategorii Electronics'",
            height=100
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_button = st.button("Analizuj", type="primary")
        with col2:
            if st.button("Wyczyść", type="secondary"):
                st.session_state.response = None
                st.session_state.code = None
                st.session_state.filtered_data = None
                st.rerun()
    
    # Jeżeli przycisk został wciśnięty i dane są dostępne
    if submit_button and df is not None and user_input:
        # Inicjalizacja zmiennych sesji, jeśli nie istnieją
        if 'response' not in st.session_state:
            st.session_state.response = None
        if 'code' not in st.session_state:
            st.session_state.code = None
        if 'filtered_data' not in st.session_state:
            st.session_state.filtered_data = None
            
        # Przygotowanie zapytania z systemowym promptem
        system_prompt = load_system_prompt()
        final_prompt = system_prompt + '\n\n' + user_input
        
        # Generowanie odpowiedzi
        response = generate_response(
            final_prompt, 
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Czyszczenie odpowiedzi z tagów think
        clean_response = remove_think_tags(response)
        
        # Zapisanie do sesji
        st.session_state.response = clean_response
        
        # Znajdź blok kodu Python
        code_match = re.search(r'```python\s*(.*?)\s*```', clean_response, re.DOTALL)
        if code_match:
            code = code_match.group(1)
            st.session_state.code = code
            
            # Wykonanie kodu
            filtered_data = execute_generated_code(code, df)
            st.session_state.filtered_data = filtered_data
        else:
            # Próba wykonania całej odpowiedzi jako kodu
            st.session_state.code = clean_response
            filtered_data = execute_generated_code(clean_response, df)
            st.session_state.filtered_data = filtered_data
    
    # Wyświetlenie wyników
    if hasattr(st.session_state, 'response') and st.session_state.response:
        # Sekcja z odpowiedzią modelu
        with st.expander("Odpowiedź modelu", expanded=True):
            st.markdown(st.session_state.response)
        
        # Sekcja z wyświetlonym kodem
        if st.session_state.code:
            with st.expander("Wygenerowany kod", expanded=True):
                st.code(st.session_state.code, language="python")
        
        # Sekcja z przefiltrowanymi danymi
        if st.session_state.filtered_data is not None:
            filtered_df = st.session_state.filtered_data
            
            st.subheader("Wyniki analizy")
            st.dataframe(filtered_df, use_container_width=True)
            
            # Link do pobrania
            st.markdown(get_csv_download_link(filtered_df), unsafe_allow_html=True)
            
            # Automatyczne tworzenie wykresu, jeśli dane mają odpowiednie kolumny
            if 'Category' in filtered_df.columns and any(col for col in filtered_df.columns if 'Sales' in col or 'Revenue' in col or 'Amount' in col):
                st.subheader("Wizualizacja danych")
                
                # Znajdź kolumnę zawierającą dane o sprzedaży
                sales_column = next((col for col in filtered_df.columns if 'Sales' in col or 'Revenue' in col or 'Amount' in col), None)
                
                if sales_column:
                    # Grupowanie i sumowanie sprzedaży wg kategorii
                    sales_by_category = filtered_df.groupby('Category')[sales_column].sum().sort_values(ascending=False)
                    
                    # Tworzenie wykresu słupkowego
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sales_by_category.plot(kind='bar', color='lightgreen', ax=ax)
                    plt.title(f'{sales_column} wg kategorii')
                    plt.xlabel('Kategoria')
                    plt.ylabel(sales_column)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.grid(axis='y', linestyle='--', alpha=0.5)
                    
                    st.pyplot(fig)

if __name__ == "__main__":
    main()