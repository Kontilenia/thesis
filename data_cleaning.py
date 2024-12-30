from datasets import load_dataset
import pandas as pd
import re
from typing import List
import requests
from bs4 import BeautifulSoup

def pattern_cleaning(
    df: pd.DataFrame,
    exceptions: List[int]
    ) -> pd.DataFrame:
    """
    Function that cleans 4 unwanted patterns from the dataset
    regarding, indexing of questions, special characters, speaker's name
    and description of questions.

    Arguments:
        df: Dataframe to be cleaned
        exceptions: exception list of indexes where the disception of the
        question is needed

    Returns:
        df: Cleaned dataframe
    """

    """
    Regex explanation:

    ^ matches the start of the string
    (\d+\.|Part \d+:|Q\d*:|\d+\. Q\d*: ) is a capturing group that
    matches one of the following:
        \d+\. : one or more digits followed by a period

        Part \d+: : the string "Part " followed by one or more digits,
        a colon, and an optional space

        Q\d*: : the string "Q" followed by one or more digits, a colon,
        and an optional space

        \d+\. Q\d*: : one or more digits followed by a period, a space,
        "Q", one or more digits, a colon, and an optional space

        - : start sentence with "-"
    """

    # 1) Remove indexing from questions
    index_pattern = r'^(\d+\. Q\d+:|\d+\.|Part \d+:|Q\d+:|-)'
    df['question'] = df['question'].str.replace(
        index_pattern,
        '',
        regex=True
        )

    # 2) Remove quotes and new line espace characters
    df['question'] = df['question'].str.replace(
        r'["\n]',
        '',
        regex=True
        )
    
    df['interview_answer'] = df['interview_answer'].str.replace(
        r'["\n\r]',
        '',
        regex=True
        )
    
    df['interview_question'] = df['interview_question'].str.replace(
        r'["\n\r]',
        '',
        regex=True
        )

    # 3) Remove first sentence from answer (indicates which present is
    # speaking)
    sentence_pattern = r'^[^.]+\.?'
    df['interview_answer'] = df['interview_answer'].str.replace(
        sentence_pattern,
        '',
        regex=True
        )

    # 4) Remove description from questions
    df.loc[~df.index.isin(exceptions), 'question'] = df.loc[
        ~df.index.isin(exceptions), 'question'].apply(
        lambda x: re.sub(r'^[^:]+: ', '', x))
    return df


def get_italic_sentences(url: str) -> list:
    """
    Function to get italic sentences from a url, optimized with error
    handling

    Arguments:
        url: Link of the text

    Returns:
        Text with italics except specific phrases
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for bad responses
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text from the <div> with class "field-docs-content"
        div_content = soup.find('div', class_='field-docs-content')

        # Return an empty list if the div is not found
        if div_content is None:
            return []

        exception_list = {
            "The President.",
            "Q.",
            "Inaudible",
            "inaudible"
            }

        # Extract unique sentences from <i> or <em> tags, excluding
        # specific phrases
        italic_sentences = {
            i.get_text(strip=True)
            for i in div_content.find_all(['i', 'em'])
            }
        
        return [
            sentence
            for sentence in italic_sentences
            if sentence not in exception_list
            ]

    except (requests.RequestException, AttributeError) as e:
        print(f"Error retrieving or parsing {url}: {e}")
        return []


def clean_interview_answer(row: pd.Series, url_sentences: set) -> str:
    """
    Remove unnecessary sentences from a interview_answer in a
    vectorized manner

    Arguments:
        row: row of a dataframe
        url_sentences: set of unique sentences to be removed
        from interview answer of a text coming from a particular
        url

    Returns:
        Interview answer string with removed sentences
    """
    unique_sentences = url_sentences.get(row['url'], [])
    interview_answer = row['interview_answer']
    for sentence in unique_sentences:
        interview_answer = interview_answer.replace(sentence, '')
    return interview_answer


def remove_unrelated_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to remove italic sentences from the 'interview_answer' column.

    Arguments:
        df: Dataframe to be cleaned

    Returns:
        df: Cleaned dataframe
    """

    # Create a dictionary to store unique sentences for each URL
    url_sentences = {}

    # Create a dictionary to store unique sentences for each URL
    unique_urls = df['url'].unique()

    # Get sentences for each URL (optionally use parallel processing for
    # speedup)
    for url in unique_urls:
        url_sentences[url] = get_italic_sentences(url)

    df['interview_answer'] = df.apply(
        lambda x: clean_interview_answer(x, url_sentences), axis=1)

    # Optional: Clean up whitespace after sentence removal
    df['interview_answer'] = df['interview_answer'].str.replace(
        r'\s+', ' ',
        regex=True
        ).str.strip()

    return df

def extra_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add inadible and multiple question labels to the dataset

    Arguments:
        df: Dataframe

    Returns:
        df: Labeled dataframe
    """
    df["inaudible"] = df['interview_answer'].str.contains('inaudible', case=False)
    df["multiple_questions"] = df['question'].str.count('\?') > 1
    df["affirmative_questions"] = ~df['question'].str.contains('\?')
    return df

def alter_class_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alter class names to match with the paper ones

    Arguments:
        df: Dataframe

    Returns:
        df: Dataframe with altered class names  

    """

    # Remove the taxonomy numbers from labels
    df["annotator1"] = df["annotator1"].str.slice(4)
    df["annotator2"] = df["annotator2"].str.slice(4)
    df["annotator3"] = df["annotator3"].str.slice(4)
    
    return df

def create_labels_train_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create labels for the dataset

    Arguments:
        df: Dataframe

    Returns:
        df: Dataframe with labels
    """

    clarity_mapping ={
    'Explicit': 'Direct Reply',
    'Implicit': 'Indirect',
    'Dodging': "Indirect",
    'Deflection': "Indirect",
    'Partial/half-answer': "Indirect",
    'General': "Indirect",
    'Contradictory': "Indirect",
    'Declining to answer': "Direct Non-Reply",
    'Claims ignorance': "Direct Non-Reply",
    'Clarification': "Direct Non-Reply",
    'Diffusion': "Indirect",
    }
    
    df["clarity_label"] = df["label"].map(clarity_mapping)
    df.rename(columns={"label": "evasion_label"}, inplace=True)
    return df

def create_labels_test_set(df: pd.DataFrame) -> pd.DataFrame:  
    """
    Create labels for the dataset

    Arguments:
        df: Dataframe

    Returns:
        df: Dataframe with labels
    """

    evasion_mapping = {
    'Direct Reply': 'Direct Reply',
    'Indirect': 'Indirect',
    'Direct Non-Reply': 'Direct Non-Reply'
    }

    df["evasion_label"] = df["label"].map(evasion_mapping)
    return df

def clean_dataset(
        df: pd.DataFrame, 
        exception_list: List[int], 
        storing_path: str, 
        full_storing_path: str, 
        column_names: List[str],
        test_set: bool = False
        ) -> None:
    """
    Cleans and saves a dataset to specified paths

    Parameters:
        df: The DataFrame to be cleaned and saved
        exception_list: A list of exception patterns to be used during pattern cleaning
        storing_path: The file path where the core cleaned dataset will be saved
        full_storing_path: The file path where the full cleaned dataset will be saved
        column_names: A list of column names to be included in the core dataset

    Returns:
        None
    """
            
    # Remove unwanted patterns
    df = pattern_cleaning(df, exception_list)

    # Extract noise from the end of interview answer
    df = remove_unrelated_text(df)

    # Add 2 more labels for multiple questions and inadible speech
    df = extra_labels(df)

    # Do extra cleaning for test set
    if test_set:
        df = alter_class_names(df)

    # Save full dataset to path
    df.to_csv(full_storing_path, index=False)

    # Get the columns necessary for training
    df = df[column_names]

    # Save core dataset to path
    df.to_csv(storing_path, index=False)

def main():
    # # Load train dataset
    # ds = load_dataset("ailsntua/QEvasion")

    # # Convert to pandas and keep only useful columns
    # df_train = ds["train"].to_pandas()

    # column_names = ["question","interview_question",
    #                  "interview_answer", "label","url"]

    # # Handpicked expeption to unwanted patterns
    # train_exception_list = [142,493,699,809,1052,1053,1446,
    #                 2417,2631,2821,3181,3390]

    # clean_dataset(
    #     df_train, train_exception_list, 
    #     "preprocessed_data/train_set.csv",
    #     "preprocessed_data/full_train_set.csv",
    #     column_names
    # )

    df_test = pd.read_csv('data/test_set.csv')

    # test column mapping
    column_mapping = {
    'Question': 'question',
    'Interview Answer': 'interview_answer',
    'link': 'url',
    "Interview Question": "interview_question",
    "Label": "label",
    "Annotator1": "annotator1",
    "Annotator2": "annotator2",
    "Annotator3": "annotator3"
    }

    df_test = df_test.rename(columns=column_mapping)

    column_names = ["question","interview_question","interview_answer", 
                    "label", "url", "annotator1", "annotator2", "annotator3"]

    # Handpicked expeption to unwanted patterns
    test_exception_list = [153, 169, 200, 300]

    clean_dataset(
        df_test, test_exception_list, 
        "preprocessed_data/test_set.csv",
        "preprocessed_data/full_test_set.csv",
        column_names,
        test_set=True)

if __name__ == "__main__":
    main()