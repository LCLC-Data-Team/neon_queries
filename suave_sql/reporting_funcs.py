import pandas as pd

import random
import pyperclip
import xlsxwriter
from collections import OrderedDict, defaultdict
from datetime import timedelta
from functools import reduce

import os
import sys 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sql_funcs
import re
import numpy as np
import ast


class Report:
    def __init__ (self, report_functions, engine_settings, start_date, end_date, report_type = 'Queries', interval = None, group_by_interval = False, projections=False):
        '''
        Creates an object that runs a report and stores its outputs

        Parameters:
            report_functions (Dict): A dictionary of functions to run. Value is 
            engine_settings (Dict): A dictionary of preferences to initialize a suave_sql object
            start_date (Str): The start date of the report period, formatted as 'YYYY-MM-DD'
            end_date (Str): The end date of the report period, formatted as 'YYYY-MM-DD'
            report_type: The sql_funcs object to use. Options include Audits, IDHS, ReferralAsks, Queries. Defaults to Queries
            interval: (optional) The date intervals within the time period to run the report over. 'MS' for month, '3MS' for quarter, 'YS' for year. Defaults to None
            group_by_interval (Bool): (optional) True groups a series of reports by time interval, False groups by each query. Defaults to False
            projections (Bool): whether to run projections for grants based on grant start/end. Defaults to False

        Examples:
            Sample report_functions::
                
                funcz = {"highest cases by team": ("highest_cases", (False,)),
                        "highest cases by atty": ("highest_cases", (True,)),
                        "all client neighborhoods":("dem_address",(False, 'region',)),
                        "custody status counts":("custody_status",(True,)),
                        "services missing staff":("service_lacks_staff",())
                            }
            
            Sample engine settings::

                engine = create_engine('mysql+pymysql://eli:password@LCLCN001/neon', isolation_level="AUTOCOMMIT")
                full_engine_settings = {
                    'engine': engine,
                    'print_SQL': True,
                    'clipboard': False,
                    'mycase': True,
                    'default_table': 'stints.neon_chd'
                }

            Create a report for all clients between January and August::

                r = Report(funcz, full_engine_settings, '2024-01-01', '2024-08-31', interval = None, report_type = 'Queries')
            
            See reports for each month between January and August side-by-side::

                r = Report(funcz, full_engine_settings, '2024-01-01', '2024-08-31', report_type = 'Queries', interval = 'MS')
            
            See reports for each quarter between January and August one at a time::

                r = Report(funcz, full_engine_settings, '2024-01-01', '2024-08-31', report_type = 'Queries', interval = '3MS', group_by_interval = True)
        '''
        self.report_functions = report_functions
        self.engine_settings = engine_settings
        self.start_date = start_date
        self.end_date = end_date
        self.report_type = report_type
        self.grant_dict = {
'a2j': {'grant_name':'A2J',
        'grant_start':'"2025-07-01"',
        'grant_end':'"2026-06-30"'},
'idhs': {'grant_name':'IDHS VP',
        'grant_start':"2025-07-01",
        'grant_end':'"2026-06-30"'},
'cvi':{'grant_name':'ICJIA - CVI',
        'grant_start':'"2025-10-01"',
        'grant_end':'"2026-09-30"'},
'r3':{'grant_name':'ICJIA - R3',
        'grant_start':'"2025-11-01"',
        'grant_end':'"2026-10-30"'},
'scan':{'grant_name':'DFSS - SCAN',
        'grant_start':'"2025-01-01"',
        'grant_end':'"2025-12-31"'},
'ryds':{'grant_name':'IDHS - RYDS',
        'grant_start':'"2025-10-01"',
        'grant_end':'"2026-09-30"'},
'jac':{'grant_name': 'JAC - CVI',
        'grant_start':'"2025-08-01"',
        'grant_end':'"2027-07-31"'}}
        self.projections = projections
        if not interval:
            self.run_a_report()
        else:
            self.group_by_interval = group_by_interval
            self.date_dict = self.make_date_dict(interval)
            self.generate_reports()

    def run_a_report(self):
        '''
        Runs the report using sql_funcs' Tables.run_report() and saves it as a dictionary. Done automatically when using the Report object.

        Example:
            To access the full dictionary of outputs::

                r.report_outputs

            To access a single query::

                r.report_outputs["all client neighborhoods"]
        '''
        full_inputs = {**{'t1':self.start_date, 't2':self.end_date}, **self.engine_settings}
        classtype = getattr(sql_funcs, self.report_type)
        q = classtype(**full_inputs)
        
        report_method_dict = {'projections':q.run_projections, 'set_interval':q.run_report}
        report_method_key = 'projections' if self.projections else 'set_interval'

        if isinstance(next(iter(self.report_functions.values())),dict) and self.projections == False:
            self.report_outputs = {}
            for sheet, queries in self.report_functions.items():
                self.report_outputs[sheet] = report_method_dict[report_method_key](func_dict = queries)
        else:
            self.report_outputs = report_method_dict[report_method_key](func_dict=self.report_functions)

    def report_to_excel(self, file_path, query_tabs = False, spacer_rows = 1):

        '''
        Saves the report outputs to an excel file

        Parameters:
            file_path: The new file path for the excel document
            query_tabs (Bool): Whether each function should get its own tab in the file. Must be False if report_functions has multiple levels. Defaults to False
            spacer_rows (Int): The number of empty rows to put between each query if not on separate tabs. Defaults to 1

        Examples:
            Save each query on separate tabs::

                r.report_to_excel(file_path="C:/Users/eli/Downloads/test.xlsx", query_tabs=True)
            
            Put two spaces between each query on the same sheet::
                
                r.report_to_excel(file_path="C:/Users/eli/Downloads/test.xlsx", spaces = True)
        '''

        def format_sheet(query_dict, sheet_name = 'Sheet1', spacer_rows = spacer_rows):
            empty.to_excel(writer,sheet_name=sheet_name, startrow=0 , startcol=0, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column('A:Z', 15, standard_format)
            row = 1
            for query_name, query_df in query_dict.items():
                print(query_name)
                column_levels = query_df.columns.nlevels
                if column_levels > 1:
                    query_df = query_df.reset_index()
                    df_start = 0 if self.projections else 1
                    query_df.iloc[df_start: , :].to_excel(writer,sheet_name=sheet_name,startrow=row, startcol=-1)
                    
                    df_spacer_rows = column_levels + spacer_rows - 1 #same space bt multiindex dfs & regulars
                else:
                    query_df.to_excel(writer,sheet_name=sheet_name,startrow=row , startcol=0, index=False)
                    df_spacer_rows = spacer_rows
                
                #index columns
                nonnum_cols = [idx for idx, col in enumerate(query_df.select_dtypes(exclude=['number']).columns)]
                if nonnum_cols:
                    if column_levels > 1:
                        worksheet.conditional_format(row + column_levels + 1, 0, row + query_df.shape[0] + column_levels, max(nonnum_cols), 
                        {'type':'no_errors', 'format':index_format})
                    else:
                        worksheet.conditional_format(row + column_levels, 0, row + query_df.shape[0] + column_levels - 1, max(nonnum_cols), 
                    {'type':'no_errors', 'format':index_format})
                # table title
                worksheet.merge_range(row-1, 0, row-1, (query_df.shape[1])-1, query_name.upper(), chart_title_format)
                row = row + len(query_df.index) + df_spacer_rows + 3
        
        def format_tabs(query_dict):
            for query_name, query_df in query_dict.items():
                empty.to_excel(writer,sheet_name=query_name,startrow=0 , startcol=0, index=False)
                worksheet = writer.sheets[query_name]
                worksheet.set_column('A:Z', 15, standard_format)

                column_levels = query_df.columns.nlevels
                if column_levels > 1:
                    query_df = query_df.reset_index()
                    query_df.iloc[1: , :].to_excel(writer,sheet_name=query_name,startrow=1 , startcol=-1)
                else:
                    query_df.to_excel(writer,sheet_name=query_name,startrow=1 , startcol=0, index=False)
                #index columns
                nonnum_cols = [idx for idx, col in enumerate(query_df.select_dtypes(exclude=['number']).columns)]
                if nonnum_cols:
                    worksheet.conditional_format(column_levels + 1, 0, query_df.shape[0] + column_levels, max(nonnum_cols),
                    {'type':'no_errors', 'format':index_format})
                #chart title
                worksheet.merge_range(0, 0, 0, (query_df.shape[1])-1, query_name.upper(), chart_title_format)

        writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
        workbook = writer.book

        #format time
        standard_format = workbook.add_format()
        standard_format.set_text_wrap()

        chart_title_format = workbook.add_format({'align': 'center'})
        chart_title_format.set_underline()

        index_format = workbook.add_format()
        index_format.set_italic()
        index_format.set_bg_color('#D7D7D7')
        index_format.set_border(1)

        empty = pd.DataFrame()

        # check if data is nested 
        random_key = random.choice(list(self.report_outputs.keys()))
        if isinstance(self.report_outputs[random_key], dict):
            for key, data_dict in self.report_outputs.items():
                format_sheet(data_dict, key)
        #cool it isnt
        else:
            if query_tabs:
                format_tabs(self.report_outputs)
            else:
                format_sheet(self.report_outputs)
        writer.close()

    def make_date_dict(self, interval):
        '''
        Makes a dictionary of dates to pass to an sql_funcs object

        Parameters:
            interval: the time interval to subset the dates. "MS" for month, "3MS" for quarter, "YS" for year
        '''

        quarters = {'January': 'Q1',
        'April':'Q2',
        'July':'Q3',
        'October':'Q4'}

        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq=interval)
        date_tally = 0

        date_dict = {}
        for date in date_range:
            date_tally = date_tally + 1

            range_start = date.strftime('%Y-%m-%d')
            if len(date_range) == date_tally:
                range_end = pd.to_datetime(self.end_date).strftime('%Y-%m-%d')
            else:
                range_end = (date_range[date_tally] - timedelta(days=1)).strftime('%Y-%m-%d')
            month_name = date.month_name()
            if interval == 'YS':
                range_name = f'{date.year}'
            elif interval == '3MS':
                range_name = f'{quarters[month_name]} {date.year}'
            else:
                range_name = f'{month_name} {date.year}'
            date_dict[range_name] = {'t1': range_start, 't2':range_end}
        self.date_dict = date_dict
        return date_dict
    
    def generate_reports(self):
        '''
        Generates reports for each time interval within the report and saves them to self.report_outputs.
    
        '''
        report_outputs = {key: OrderedDict() for key in self.report_functions} if not self.group_by_interval else {key: OrderedDict() for key in self.date_dict}
        for date_key, first_two_inputs in self.date_dict.items():
            full_inputs = {**first_two_inputs, **self.engine_settings}
            
            classtype = getattr(sql_funcs, self.report_type)
            q = classtype(**full_inputs)
            interval_report = q.run_report(func_dict=self.report_functions)
            
            for func_name, func_df in interval_report.items():
                if self.group_by_interval:
                    report_outputs[date_key][func_name] = func_df
                else:
                    report_outputs[func_name][date_key] = func_df
        if self.group_by_interval:
            self.report_outputs = report_outputs
        else:
            self.report_outputs = self.consolidate_query_outputs(report_outputs)
        return report_outputs



    def consolidate_query_outputs(self, query_dict):
        '''
        Consolidates outputs for a given query over multiple intervals into one df

        Parameters:
            query_dict: the dictionary of report outputs
        '''
        
        def one_row(query_dict):
            df_list = []
            for key, df in query_dict.items():
                df.index = [key] * len(df)
                df = df.T
                df = df.reset_index()
                df_list.append(df)
            merged_df = reduce(lambda x, y: x.merge(y, on='index'), df_list)
            return merged_df
        
        def one_numeric(query_dict, indices, numeric_cols):
            df_list = [df.rename(columns={numeric_cols[0]: key}) for key, df in query_dict.items()]
            merged_df = reduce(lambda x, y: x.merge(y, on=indices, how='outer'), df_list)
            return merged_df
        
        def many_numeric(query_dict, indices, numeric_cols):
            dfs_with_multiindex = {key: df.copy() for key, df in query_dict.items()}
            for key in dfs_with_multiindex:
                df = dfs_with_multiindex[key]
                dfs_with_multiindex[key] = df.set_index(indices)
                dfs_with_multiindex[key].columns = pd.MultiIndex.from_product([[key], numeric_cols])

            merged_df = pd.concat(dfs_with_multiindex.values(), axis=1)
            return merged_df
        
        output_dict = OrderedDict()
        for query_name, query_dict in query_dict.items():
            random_key = random.choice(list(query_dict.keys()))
            random_df = query_dict[random_key]
            if len(random_df) == 1:
                out_df = one_row(query_dict)
            else:
                indices = random_df.select_dtypes(exclude=['number']).columns.tolist()
                numeric_cols = random_df.select_dtypes(include=['number']).columns.tolist()
                if len(numeric_cols) == 1:
                    out_df = one_numeric(query_dict, indices, numeric_cols)
                else:
                    out_df = many_numeric(query_dict, indices, numeric_cols)
            
            output_dict[query_name] = out_df
        return output_dict

class ReportFromXlsxTemplate:
    def __init__(self, excel_file_path, t1=None, t2=None,engine=None, include_grants = [], include_tabs = []):  
        '''
        Process a Report Template file and run reports for each grant represented in the file.
        t1, t2, and engine can be set to None for the purpose of reformatting an excel sheet exclusively. 
        The formatted df is preserved as .base_df, the dictionary of outputs/queries is preserved as .query_dict
        
        Parameters:
            excel_file_path: path to your completed Report Template
            t1: start date of report period, formatted as 'YYYY-MM-DD'. 
            t2: end date of report period, formatted as 'YYYY-MM-DD'
            engine: mysql engine
            include_grants (optional): subset of grants within the excel file to include. Can include 'all' for non-grant requests
            include_tabs (optional): subset of tabs/report types to include (ie: PPR, audit). 
        ''' 
        self.alias_dict = {'r3':['ICJIA R3','ICJIA - R3', 'R3', 'r3'],
              'cvi':['ICJIA CVI', 'ICJIA - CVI', 'CVI', 'cvi'],
              'idhs':['IDHS VP','IDHS - VP', 'VP', 'vp'],
              'ryds':['IDHS RYDS', 'IDHS - RYDS', 'RYDS', 'ryds'],
              'jac':['JAC CVI', 'JAC - CVI', 'JAC', 'jac'],
              'all':['All',None, '', 'all']}     
        narrative_df = pd.read_excel(excel_file_path, sheet_name="Narrative Key")
        self.format_narrative = True if len(narrative_df) > 0 else False
        narrative_df = narrative_df.dropna()
        narrative_df['narrative_id'] = narrative_df['grant_name'].map({v: k for k, lst in self.alias_dict.items() for v in lst}) + "_" + narrative_df['narrative_number'].astype(int).astype(str)
        self.narrative_df = narrative_df

        df = pd.read_excel(excel_file_path, sheet_name="Methods")
        self.base_df = self.format_df(df)

        # filter out grants to be included 
        include_grants = include_grants if isinstance(include_grants, list) else [include_grants]
        self.include_tabs = include_tabs if isinstance(include_tabs, list) else [include_tabs]
        print(self.include_tabs)
        self.filter_query_df(include_grants, self.include_tabs)

        self.base_df = self.parse_df(self.base_df)
        self.format_dictionary()
        if t1 and t2 and engine:
            self.generate_output_dictionary(t1, t2, engine)
        self.output_dict = self.format_output_dictionary(self.query_dict)

    def filter_query_df(self, grant_list, tab_list):
        '''
        Subset the df to only include grants found in grant_list

        Parameters:
            grant_list: list of grants to include
        '''
        if len(grant_list)> 0:
            found_keys = []
            for key, value in self.alias_dict.items():
                for grant in grant_list:
                    if grant in value:
                        found_keys.append(key)
            self.base_df = self.base_df[self.base_df['row_id'].str.startswith(tuple(found_keys))]
        
        if len(tab_list)> 0:
            temp_df = self.base_df.copy()
            temp_df['where_used'] = temp_df['where_used'].apply(
            lambda x: [item for item in x if item in set(tab_list)])
            
            temp_df = temp_df[temp_df.where_used.str.len() > 0 ]
            print(temp_df)
            self.base_df = temp_df

    def format_df(self, df):
        df['grant_name'] = df['grant_name'].fillna('all')
        df['where_used'] = df['where_used'].str.split(', ')
        df['row_id'] = df.groupby('grant_name').cumcount() + 1
        df['row_id'] = df['grant_name'].map({v: k for k, lst in self.alias_dict.items() for v in lst}) + "_" + df['row_id'].astype(str)
        return df

    def parse_df(self, df):
        '''
        Reformat base_df to parse numbers in _row/_number cols, extract formatted methods from method col, and generate row_id for each row

        Parameters:
            df: dataframe of the excel file
        '''

        def parse_method(meth):
            """Parse a string representing a sql_query call into components suitable for the Report object."""
            if meth.startswith('.'):
                #print(meth)
                meth = meth[1:]  # remove leading dot
                query_name, vars_str = meth.split("(", 1)
                vars_str = vars_str.rsplit(")", 1)[0].strip()
                args_out = []
                kwargs_out = {}
                if vars_str:
                    # function has variables
                    if (vars_str.startswith(("f'", 'f"', "f'''", 'f"""'))):
                        # keep fstring formatted as-is. 
                        # breaks in an instance where you need an fstring and another argument. rewrite code when time comes
                        # actually, no fstrings work rn but I am Annoyed. 
                        args_out.append(vars_str)
                    
                    else:
                        # cool no fstring
                        parts = vars_str.split(",")

                        for part in parts:
                            part = part.strip()
                            if "=" in part:
                                # works for kwargs
                                key, value = part.split("=", 1)
                                key = key.strip()
                                value = ast.literal_eval(value.strip())
                                kwargs_out[key] = value
                            else:
                                # just append the values in order
                                args_out.append(ast.literal_eval(part))

                return query_name, tuple(args_out), kwargs_out
            else:
                return meth, (), {}

        def parse_value(val):
            "Convert a single string value to an appropriate Python literal."
            # Case 1: actual list or array
            if isinstance(val, (list, tuple)):
                parts = [str(v).strip() for v in val]

            # Case 2: empty or null
            elif val is None or (isinstance(val, float) and np.isnan(val)):
                return np.nan

            # Case 3: string
            else:
                s = str(val).strip()
                if s == "":
                    return np.nan

                # If string looks like list literal:  "['18','20-22']"
                if s.startswith("[") and s.endswith("]"):
                    s = s[1:-1].replace("'", "").replace('"', "")
                parts = [p.strip() for p in s.split(",") if p.strip()]

            result = []
            for part in parts:
                if re.match(r"^\d+-\d+$", part):  # range
                    start, end = map(int, part.split("-"))
                    result.extend(range(start, end + 1))
                elif part.isdigit():
                    result.append(int(part))
                # ignore anything else

            return result if result else np.nan
        df["report_row"] = df["report_row"].apply(parse_value)
        df['narrative_number'] = df['narrative_number'].apply(parse_value)
        df['formatted_method'] = df['method'].apply(parse_method)
        #print(df)
        return df
    
    def format_dictionary(self):
        '''
        '''
        def convert_df_to_dict(df):
            query_dict = {}
            for _, row in df.iterrows():
                sub_dict = {}
                for col in ['where_used','method','metric','comments','report_row','narrative_number','formatted_method']:
                    val = row[col]
                    if isinstance(val, (float, int)) and pd.isna(val):
                        continue
                    sub_dict[col] = val
                query_dict[row['row_id']] = sub_dict
            return query_dict
        
        def format_dict_for_report(query_dict):
            outer_dict = { k: {} for k in set([key.split('_')[0] for key in list(query_dict.keys())])}
            for k, v in query_dict.items():
                outer_dict[k.split('_')[0]][k] = v['formatted_method']
            return outer_dict
        
        query_dict = convert_df_to_dict(self.base_df)
        outer_dict = format_dict_for_report(query_dict)
        self.formatted_report_dict = outer_dict
        self.query_dict = query_dict
        print('dictionary reformatted as .formatted_report_dict')
        return('dictionary reformatted as .formatted_report_dict')

    def generate_output_dictionary(self, t1, t2,engine, function_dict = None, default_table = None):
        '''
        when function_dict = None, self.formatted_report_dict is used
        '''
        def actually_run_report(outer_dict, default_table):
            output_dict = {}
            for grant_name, grant_funcs in outer_dict.items():
                standard_inputs = {
                    'engine': engine,'print_SQL': False,'clipboard': False,'mycase': True,
                    'default_table': default_table}
                funcz = grant_funcs

                if grant_name == 'all':
                    r = Report(funcz, standard_inputs, t1, t2, interval = None, report_type = 'Queries')
                else:
                    standard_inputs['grant_type'] = grant_name
                    standard_inputs['default_table'] = 'stints.neon'
                    r = Report(funcz, standard_inputs, t1, t2, interval = None, report_type = 'Grants')
                
                output_dict[grant_name] = r.report_outputs
            return output_dict
        
        def add_outputs_to_query_dict(output_dict, full_query_dict):
            for full_grant_dict in output_dict.values():
                for k, v in full_grant_dict.items():
                    if isinstance(v, pd.DataFrame):
                        full_query_dict[k]['output'] = v
                    else:
                        #print(v)
                        full_query_dict[k]['output'] = full_query_dict[k]['formatted_method'] if v is None or v.startswith('Error') else v
        
        if not function_dict:
            function_dict = self.formatted_report_dict
        default_table = "stints.neon" if not default_table else default_table
        output_dict =  actually_run_report(function_dict, default_table)
        add_outputs_to_query_dict(output_dict, self.query_dict)
        return self.query_dict
    
    def format_output_dictionary(self, query_dict):
        def group_and_sort(data):
            ## place all dictionary entries in order of first row in report
            # ---- Group by prefix ----
            prefix_groups = defaultdict(dict)
            for key, item in data.items():
                prefix = key.split("_", 1)[0]
                prefix_groups[prefix][key] = item

            result = {}
            keys_to_keep = ['method','metric','comments', 'output']
            for prefix, items in prefix_groups.items():
                # ---- Group by where_used ----
                where_used_groups = defaultdict(dict)
                for key, item in items.items():
                    for wu in item.get("where_used", []):
                        temp_keys=keys_to_keep.copy()
                        if wu == 'Narrative':
                            temp_keys.append('narrative_number')
                        else:
                            temp_keys.append('report_row')
                        where_used_groups[wu][key] = {k: item[k] for k in temp_keys if k in item}

                # ---- Sort each A/B/C dictionary ----
                sorted_groups = {}

                for wu, entries in where_used_groups.items():
                    if wu == "Narrative":
                        # sort by first element of B_num
                        #print(entries)
                        sorted_keys = sorted(
                            entries.keys(),
                            key=lambda k: entries[k]["narrative_number"][0]
                        )
                    
                    else:
                        try:
                            #print(entries)
                            # sort by first element of _num
                            sorted_keys = sorted(
                                entries.keys(),
                                key=lambda k: entries[k]["report_row"][0])
                        except KeyError:
                            sorted_keys = sorted(entries.keys())

                    # rebuild dictionary in sorted order
                    sorted_groups[wu] = {k: entries[k] for k in sorted_keys}

                result[prefix] = sorted_groups

            return result
        
        def sort_narrative_dict(narrative_dict, narrative_key_df = None):
            # regroup narrative dictionary to following format
            # narrative prompt: {narrative_info: {where_used = str, previous_answer = str},
            #                    {query_entry_1}, {query_entry_2}}
            def initial_sort():
                exploded = defaultdict(dict)
                for key, item in narrative_dict.items():
                    for num in item.get("narrative_number", []):
                        exploded[f'{key.split("_", 1)[0]}_{num}'][key] = item
                return dict(sorted(exploded.items()))

            initial = initial_sort()
            renamed = {}
            for k, v in initial.items():
                row = narrative_key_df[narrative_key_df['narrative_id']== k]
                question_name = row['narrative_question'].item()
                where_used = row['where_used'].item()
                previous_answer = row['previous_answer'].item()
                renamed[question_name] = v
                renamed[question_name]['narrative_info'] = {
                    'where_used': where_used,
                    'previous_answer': previous_answer}

            return renamed 

        def reformat_all_narratives(sorted_dictionary, narrative_key_df):
            sorted_dictionary_copy = sorted_dictionary.copy()
            for grant_name, grant_dict in sorted_dictionary.items():
                if 'Narrative' in grant_dict:
                    sub_dictionary = grant_dict['Narrative']
                    new_narrative_dict = sort_narrative_dict(sub_dictionary, narrative_key_df)
                    sorted_dictionary_copy[grant_name]['Narrative'] = new_narrative_dict
            return sorted_dictionary_copy
        

        grouped_sorted = group_and_sort(query_dict)
        grouped_sorted_narr = reformat_all_narratives(grouped_sorted, self.narrative_df)

        # flattened_output_dict used 4 shiny
        self.flattened_output_dict = {(in_k if u == 'all' else f"{self.alias_dict[u][1]}: {in_k}"): in_v
                for u, inner_dict in grouped_sorted_narr.items()
                for in_k, in_v in inner_dict.items()}

        return grouped_sorted_narr
    
    def find_missing_outputs(self, flattened_key):
        needed_queries = []
        missing_candidates = self.flattened_output_dict[flattened_key]
        for lone_query_key, lone_query_dict in missing_candidates.items():
            #print(lone_query_dict)
            if 'output' not in lone_query_dict:
                needed_queries.append(lone_query_key)
        
        needed_outputs = {k: inner_match 
            for k, v in self.formatted_report_dict.items() 
            if (inner_match := {ik: iv for ik, iv in v.items() if ik in needed_queries})}
       
        return needed_outputs


    def add_missing_outputs(self, flattened_key, engine, t1, t2, default_table=None):
        missing_outputs = self.find_missing_outputs(flattened_key)
        if missing_outputs:
            self.generate_output_dictionary(t1, t2, engine, missing_outputs, default_table)
            self.output_dict = self.format_output_dictionary(self.query_dict)


    
    def report_to_excel(self, file_path, spacer_rows = 1, include_method = True,
                        format_dict = {
                        'standard': {'text_wrap': True},
                        'chart_title':{'align': 'center', 'bold':True, 'bg_color':'#DAEFF5','text_wrap':True,'underline':True},
                        'index':{'italic':True, 'bg_color':'#D7D7D7', 'border':1},
                        'query_info':{'italic':True},
                        'query_info_comments':{'text_wrap':False},
                        'narrative_title':{'align': 'center', 'bold':False, 'bg_color':'#51B4CF','text_wrap':True}
                    }):

        def format_narrative_info_dict(narrative_info_dict, worksheet, row):
            for k, v in narrative_info_dict.items():
                worksheet.set_row(row, None, info_comments_format)
                worksheet.write(row, 0, f"{k}:", info_format)
                worksheet.write(row, 1, v)
                row += 1
            return row

        def format_lone_output(query_name, query_contents, worksheet, row):
            query_df = query_contents.get("output")
            query_title = query_contents.get("metric")
            row_number = query_contents.get("report_row")
            comments = query_contents.get("comments")
            method = query_contents.get("method")

            # add title
            if isinstance(query_df, pd.DataFrame):
                worksheet.merge_range(row, 0, row, (query_df.shape[1])-1, query_title.upper(), chart_title_format)
            else:
                worksheet.merge_range(row, 0, row, 2, query_title.upper(), chart_title_format)
            row += 1

            if row_number:
                worksheet.set_row(row, None, info_comments_format)
                worksheet.write(row, 0, "Row Number(s):", info_format)
                worksheet.write(row, 1, ', '.join([str(num) for num in row_number]), info_comments_format)
                row += 1

            
            if comments:
                worksheet.set_row(row, None, info_comments_format)
                worksheet.write(row, 0, "Comments:", info_format)
                worksheet.write(row, 1, comments)
                row += 1
            
            if include_method and method:
                worksheet.set_row(row, None, info_comments_format)
                worksheet.write(row, 0, "Neon_Queries Method:", info_format)
                worksheet.write(row, 1, method)
                row += 1
            
            if isinstance(query_df, pd.DataFrame):
                #print(query_name)
                column_levels = query_df.columns.nlevels
                if column_levels > 1:
                    query_df = query_df.reset_index()
                    df_start = 1
                    query_df.iloc[df_start: , :].to_excel(writer,sheet_name=sheet_name,startrow=row, startcol=-1)
                    
                    df_spacer_rows = column_levels + spacer_rows - 1 #same space bt multiindex dfs & regulars
                if isinstance(query_df.columns, pd.RangeIndex):
                    query_df.reset_index()
                    query_df.to_excel(writer,sheet_name=sheet_name,startrow=row , startcol=0, index=True)
                    df_spacer_rows = spacer_rows

                else:
                    if query_df.shape == (1, 1):
                        query_df.reset_index()
                    query_df.to_excel(writer,sheet_name=sheet_name,startrow=row , startcol=0, index=False)
                    df_spacer_rows = spacer_rows
                
                #index columns
                nonnum_cols = [idx for idx, col in enumerate(query_df.select_dtypes(exclude=['number']).columns)]
                if nonnum_cols:
                    if column_levels > 1:
                        worksheet.conditional_format(row + column_levels + 1, 0, row + query_df.shape[0] + column_levels, max(nonnum_cols), 
                        {'type':'no_errors', 'format':index_format})
                    else:
                        worksheet.conditional_format(row + column_levels, 0, row + query_df.shape[0] + column_levels - 1, max(nonnum_cols), 
                    {'type':'no_errors', 'format':index_format})
                # table title
                row = row + len(query_df.index) + df_spacer_rows + 2 + spacer_rows
            else:
                worksheet.write(row, 0, str(query_df))
                row = row + 4 + spacer_rows
            return row

        
        def format_sheet(query_dict, sheet_name = 'Sheet1', spacer_rows = spacer_rows):
            empty.to_excel(writer,sheet_name=sheet_name, startrow=0 , startcol=0, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column('A:Z', 18, standard_format)
            row = 1
            if sheet_name.endswith('Narrative') or self.include_tabs == ['Narrative']:
                #print(sheet_name)
                #print(query_dict)
                for narrative_name, narrative_dict in query_dict.items():
                    worksheet.merge_range(row, 0, row, 3, narrative_name.upper(), narrative_title_format)
                    row += 1
                    narrative_info_dict = narrative_dict.pop('narrative_info', None)
                    if narrative_info_dict:
                        row = format_narrative_info_dict(narrative_info_dict, worksheet, row)
                    #print(narrative_name, narrative_dict, narrative_info_dict)
                    for query_name, query_contents in narrative_dict.items():
                        #print(query_name)
                        row = format_lone_output(query_name, query_contents, worksheet, row)
                        #print(row)

            else:
                for query_name, query_contents in query_dict.items():
                    print(query_name, query_contents)
                    row = format_lone_output(query_name, query_contents, worksheet, row)
            
        def count_sheet_names(output_dict):
            all_keys = set()
            for sub_dict in output_dict.values():
                if isinstance(sub_dict, dict):
                    all_keys.update(sub_dict.keys())
            return len(all_keys)

        

        if file_path.endswith('xlsx'):
            # making one excel file, everyone is happy
            writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
            workbook = writer.book

            #format time
            standard_format = workbook.add_format(format_dict['standard'])
            chart_title_format = workbook.add_format(format_dict['chart_title'])
            index_format = workbook.add_format(format_dict['index'])
            info_format = workbook.add_format(format_dict['query_info'])
            info_comments_format = workbook.add_format(format_dict['query_info_comments'])
            narrative_title_format = workbook.add_format(format_dict['narrative_title'])

            # check if there's only one sheet name for each grant. (ie: everything is 'audit')
            # if so, remove document key from the sheet name
            num_sheet_names = count_sheet_names(self.output_dict)

            empty = pd.DataFrame()
            for grant_name, grant_dict in self.output_dict.items():
                for document_key, data_dict in grant_dict.items():
                    print(grant_name)
                    
                    # take the first item in alias dict as the formalized grant name
                    formal_grant_name = self.alias_dict[grant_name][0]
                    sheet_name = formal_grant_name + " " + document_key if num_sheet_names > 1 else formal_grant_name
                    sheet_name = sheet_name[:30]
                    format_sheet(data_dict, sheet_name, spacer_rows)


            writer.close()
        
        else:
            # create folder if doesn't exist
            os.makedirs(file_path, exist_ok=True)
            for grant_name, grant_dict in self.output_dict.items():
                formal_grant_name = self.alias_dict[grant_name][0]
                file_name = f'{formal_grant_name}.xlsx'
                #print(file_name)
                excel_file = os.path.join(file_path, file_name)

                writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')
                workbook = writer.book

                #format time
                standard_format = workbook.add_format(format_dict['standard'])
                chart_title_format = workbook.add_format(format_dict['chart_title'])
                index_format = workbook.add_format(format_dict['index'])
                info_format = workbook.add_format(format_dict['query_info'])
                info_comments_format = workbook.add_format(format_dict['query_info_comments'])
                narrative_title_format = workbook.add_format(format_dict['narrative_title'])
                
                empty = pd.DataFrame()
                for document_key, data_dict in grant_dict.items():
                    sheet_name = document_key
                    sheet_name = sheet_name[:30]
                    format_sheet(data_dict, sheet_name, spacer_rows)
                writer.close()