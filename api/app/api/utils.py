import os
import openai
from typing import List, Dict
from sqlalchemy import text
from ..config import OPENAI_KEY
from ..config import csv_files, UPLOAD_FILE_DEST
from collections import OrderedDict
import pandas as pd
from pandasql import sqldf
from io import StringIO
import sqlparse
import joblib

openai.api_key = OPENAI_KEY

GRAPH_PROMPT = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant that replies only in JSON and gives the best graph types to visualize a dataset"
            "\n"
            "Given a table description, you will reply in the following form:"
            "[{{'graph':'type','x_string':'table_column_name','y_string':'table_column_name'}}...]"
            "The following is the of table description of the table you need to generate a graph description for:\n"
            "---------------------\n"
            "{table_schema}"
            "---------------------\n"
        )
    }
]

MESSAGE_PROMPT = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant for generating syntactically correct read-only SQL to answer a given question by selecting certain columns from an SQL table."
            "\n"
            "The following is the of table you can query:\n"
            "---------------------\n"
            "{table_schema}"
            "---------------------\n"
        )
    }
]

MSG_WITH_SCHEMA_AND_WARNINGS = (
    "Generate syntactically correct read-only SQL to answer the following question/command: {natural_language_query}"
    "The following are schemas of tables you can query:\n"
    "---------------------\n"
    "{table_schema}"
    "---------------------\n"
)

MSG_WITH_ERROR_TRY_AGAIN = (
    "Try again. "
    "The SQL query you just generated resulted in the following error message:\n"
    "{error_message}"
)


def is_read_only_query(sql_query: str):
    """
    Checks if the given SQL query string is read-only.
    Returns True if the query is read-only, False otherwise.
    """
    # List of SQL statements that modify data in the database
    modifying_statements = ["INSERT", "UPDATE", "DELETE",
                            "DROP", "CREATE", "ALTER", "GRANT", "TRUNCATE"]

    # Check if the query contains any modifying statements
    for statement in modifying_statements:
        if statement in sql_query.upper():
            return False

    # If no modifying statements are found, the query is read-only
    return True


class NotReadOnlyException(Exception):
    pass


def get_assistant_message(temperature: int = 0, model: str = "gpt-3.5-turbo", messages: List[Dict[str, str]] = MESSAGE_PROMPT) -> str:
    res = openai.ChatCompletion.create(
        model=model, temperature=temperature, messages=messages)
    assistant_message = res['choices'][0]
    return assistant_message


def execute_sql(sql_query: str):
    if not is_read_only_query(sql_query):
        raise NotReadOnlyException("Only read-only queries are allowed.")

    df = pd.read_parquet(os.path.join(
        UPLOAD_FILE_DEST, 'uploaded_file.parquet'))
    result_df = sqldf(sql_query, locals())
    # convert the dataframe to a CSV string
    result_df_csv_string = StringIO()

    # This saves the result_df to the string buffer passed in on the first argument
    result_df.to_csv(result_df_csv_string)

    filter_tokens = []
    parsed = sqlparse.parse(sql_query)[0]
    for token in parsed.tokens:
        if isinstance(token, sqlparse.sql.TokenList) and token.token_first().value in ['WHERE', 'LIKE', 'IN', 'BETWEEN']:
            filter_tokens.append(token.normalized)

    return {
        'column_names': list(result_df.columns),
        'csv_result': result_df_csv_string.getvalue(),
        'filter_tokens': filter_tokens,
        'result_df': result_df
    }


def generate_table_schema_string(summary_df=None):
    if summary_df is None:
        summary_df = pd.read_parquet(os.path.join(
            UPLOAD_FILE_DEST, 'summary_file.parquet'))

    categorical_column_info = "Column Name: {column_name}, Data Type: {dtype}, Total Count: {count}, Unique Count: {unique}"
    numerical_column_info = "Column Name: {column_name}, Data Type: {dtype}, Total Count: {count}, min: {min_value:.2f}, max: {max_value:.2f}, mean: {mean_value:.2f}, std: {std:.2f}"

    output_string = "Table name: 'df'\n"
    for column in summary_df.columns:
        total_count = summary_df[column]["count"]
        unique_count = None
        if "unique" in summary_df[column]:
            unique_count = summary_df[column]["unique"]
        column_dtype = summary_df[column]["dtype"]
        min_value = summary_df[column]["min"]
        if not pd.isna(min_value):
            min_value = float(min_value)

        max_value = summary_df[column]["max"]
        if not pd.isna(max_value):
            max_value = float(max_value)

        mean_value = summary_df[column]["mean"]
        if not pd.isna(mean_value):
            mean_value = float(mean_value)

        std_value = summary_df[column]["std"]
        if not pd.isna(std_value):
            std_value = float(std_value)

        if column_dtype == "string":
            output_string += categorical_column_info.format(
                column_name=column, dtype=column_dtype, count=total_count, unique=unique_count) + "\n"
        else:
            output_string += numerical_column_info.format(column_name=column, dtype=column_dtype, count=total_count,
                                                          min_value=min_value, max_value=max_value, mean_value=mean_value, std=std_value)
        output_string += "\n"

    return output_string


def format_system_prompt_with_table_schema(message_to_format, table_schema):
    output_messages = []
    for message in message_to_format:
        if "content" in message:
            formatted_str = message["content"].format(
                table_schema=table_schema)
            message['content'] = formatted_str
            output_messages.append(message)
        else:
            output_messages.append(message)
    return output_messages


def text_to_sql_with_retry(natural_language_query, k=3):
    """
    Tries to take a natural language query and generate valid SQL to answer it K times
    """
    table_schema = generate_table_schema_string()

    content = MSG_WITH_SCHEMA_AND_WARNINGS.format(
        natural_language_query=natural_language_query, table_schema=table_schema)
    messages = format_system_prompt_with_table_schema(MESSAGE_PROMPT, table_schema)
    messages.append({
        "role": "user",
        "content": content
    })

    assistant_message = None
    for _ in range(k):
        assistant_message = get_assistant_message(messages=messages)
        sql_query = _clean_message_content(
            assistant_message['message']['content'])

        response = execute_sql(sql_query)
        if response:
            result_df = response['result_df'].reset_index()
            del response["result_df"]
            summary_df = generate_summary_dataframe(result_df)
            table_schema_string = generate_table_schema_string(summary_df=summary_df)
            messages = format_system_prompt_with_table_schema(GRAPH_PROMPT, table_schema_string)
            assistant_message = get_assistant_message(messages=messages)
            response['graph'] = assistant_message['message']['content']

            # Generated SQL query did not produce exception. Return result
            return response, sql_query

    print("Could not generate SQL query after {k} tries.".format(k=k))
    return None, None


def generate_summary_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    describe_df = df.describe(include='all')

    # Iterate through all columns and try to infer the data type
    dtype = {column: str(pd.api.types.infer_dtype(
        df[column])) for column in df.columns}

    # Generate a DataFrame from the item
    dtype_df = pd.DataFrame.from_dict(dtype, orient='index', columns=["dtype"])

    # Create a unified DataFrame between these two items, save as string as parquet does not support mixed types
    summary_df = pd.concat([describe_df, dtype_df.T]).astype('string')

    return summary_df


def _clean_message_content(assistant_message_content):
    """
    Cleans message content to extract the SQL query
    """
    # Ignore text after the SQL query terminator `;`
    assistant_message_content = assistant_message_content.split(";")[0]

    # Remove prefix for corrected query assistant message
    split_corrected_query_message = assistant_message_content.split(":")
    if len(split_corrected_query_message) > 1:
        sql_query = split_corrected_query_message[1].strip()
    else:
        sql_query = assistant_message_content
    return sql_query
