import requests
import json
import pandas as pd
import re
import speech_recognition as sr
import matplotlib.pyplot as plt

with open("instruction\prompt_instruction.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# Ścieżka do Twojego pliku
csv_path = "datasets\\amazon_sales_data 2025.csv"


def remove_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate_response(prompt, model_name="deepseek-r1:14b", max_tokens=500, temperature=0.7):
    """
    Generuje odpowiedź z modelu Deepseek zainstalowanego przez Ollama.
    
    Args:
        prompt (str): Zapytanie do modelu
        model_name (str): Nazwa modelu zainstalowanego w Ollama
        max_tokens (int): Maksymalna liczba tokenów w odpowiedzi
        temperature (float): Temperatura generowania (0.0-1.0)
        
    Returns:
        str: Wygenerowana odpowiedź
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
        response = requests.post(url, json=payload)
        
        # Sprawdzenie odpowiedzi
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "Brak odpowiedzi od modelu")
        else:
            return f"Błąd API: {response.status_code}, {response.text}"
            
    except Exception as e:
        return f"Wystąpił błąd: {str(e)}"

def chat_with_model(model_name="deepseek-r1:14b"):
    """
    Prowadzi interaktywną konwersację z modelem.
    
    Args:
        model_name (str): Nazwa modelu zainstalowanego w Ollama
        Prowadzi interaktywną konwersację z modelem i wykonuje kod na DataFrame.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Nie udało się wczytać pliku CSV: {e}")
        return

    print(f"Rozpoczynanie czatu z modelem {model_name}. Wpisz 'exit' aby zakończyć.")
    
    while True:
        user_input = input("\nUser_Instruction: ")
        #user_input = get_voice_input()
        if user_input.lower() in ["koniec", "exit", "q", "koniec_1"]:
            print("Koniec konwersacji.")
            break
            
        final_prompt = system_prompt + '//n'+ user_input
        response = generate_response(final_prompt, model_name=model_name)
        clean_code = remove_think_tags(response)
        print(clean_code)

        # Wykonanie kodu z osobnej funkcji
        filtering_data = execute_generated_code(clean_code, df)
        print(filtering_data)

# Grupowanie i sumowanie sprzedaży wg kategorii
        sales_by_category = filtering_data.groupby('Category')['Total Sales'].sum().sort_values(ascending=False)

# Tworzenie wykresu słupkowego
        plt.figure(figsize=(10, 6))
        sales_by_category.plot(kind='bar', color='lightgreen')
        plt.title('Sprzedaż wg kategorii')
        plt.xlabel('Kategoria')
        plt.ylabel('Łączna sprzedaż')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()


def execute_generated_code(code: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Wykonuje kod na przekazanym DataFrame i zwraca wynik filtrowania.

    Args:
        code (str): Kod wygenerowany przez model.
        df (pd.DataFrame): DataFrame z danymi.

    Returns:
        pd.DataFrame: Przefiltrowany DataFrame lub None.
    """
    try:
        local_vars = {"df": df}
        exec(code, {}, local_vars)

        # Szukamy potencjalnej zmiennej z wynikiem filtrowania
        for var_name in ["filtered", "filtered_df", "result"]:
            if var_name in local_vars and isinstance(local_vars[var_name], pd.DataFrame):
                return local_vars[var_name]

        print("⚠️ Nie znaleziono zmiennej z przefiltrowanym DataFrame.")
        return None

    except Exception as exec_error:
        print(f"\n❌ Błąd wykonania kodu: {exec_error}")
        return None

def get_voice_input() -> str:
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 0.5
    mic = sr.Microphone()

    print("🎙️ Mów teraz...")
    
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source,timeout=5, phrase_time_limit=20)
    
    except sr.WaitTimeoutError:
            print("⏱️ Nie rozpoczęto mówienia – spróbuj jeszcze raz.")
            return ""

    try:
        text = recognizer.recognize_google(audio, language="pl-PL")
        print(f"🗣️ Rozpoznano: {text}")
        return text
    except sr.UnknownValueError:
        print("🤷‍♂️ Nie rozpoznano mowy.")
        return ""
    except sr.RequestError as e:
        print(f"❌ Błąd rozpoznawania: {e}")
        return ""


if __name__ == "__main__":
    # Przykład prostego użycia
    model = "deepseek-r1:14b"  # Twój zainstalowany model

    chat_with_model(model_name=model)