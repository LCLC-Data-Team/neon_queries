import pandas as pd
import os
from sqlalchemy import create_engine, text, types, bindparam
import re
import pyperclip
from datetime import datetime, timedelta, date
from functools import wraps
import numpy as np


def clipboard_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        
        if isinstance(result, pd.DataFrame) and self.clipboard:
            result.to_clipboard(index=False)
        return result
    return wrapper

class Tables: 
    def __init__ (self, t1, t2, engine, print_SQL = True, 
    clipboard = False, default_table = "stints.neon", mycase = True, cloud_run=False):
        '''
        establishes settings and runs stints for the desired time period

        Parameters: 
            t1: start date, formatted as "YYYY-MM-DD"
            t2: end date, formatted as "YYYY-MM-DD"
            print_sql(Bool): whether to print the SQL statements when run, defaults to True
            clipboard(Bool): whether to copy the output table to your clipboard, defaults to False
            default_table: the source table to run queries on. defaults to "stints.neon", can also use "stints.neon_chd", or a participants table
            mycase(Bool): whether the user has a formatted MyCase SQL database, defaults to True
        
        Examples:
            Set up a table of all clients in Neon in 2024 ::
            
                e = Queries(t1 = '2024-01-01', t2='2024-12-31')
            
            Set up a table of CHD clients in Q3 of 2024::

                e = Queries(t1= '2024-07-01', t2= '2024-09-30', default_table = stints.neon_chd)


        '''
        self.engine = engine
        self.t1 = t1
        self.t2 = t2
        self.q_t1 = f"'{self.t1}'"
        self.q_t2 = f"'{self.t2}'"
        self.table = default_table
        self.mycase = mycase

        self.print_SQL = print_SQL
        self.clipboard = clipboard
        
        self.con = self.engine.connect().execution_options(autocommit=True)
        self.stints_cloud_run() if cloud_run else self.stints_run()
        self.grant_dict = {
            'a2j': {'grant_name':'A2J',
                    'grant_start':'"2025-07-01"',
                    'grant_end':'"2026-06-30"'},
            'idhs': {'grant_name':'IDHS VP',
                    'grant_start':'"2025-07-01"',
                    'grant_end':'"2026-06-30"'},
            'idhs_r':{'grant_name':'IDHS - R',
                    'grant_start':'"2024-07-01"',
                    'grant_end':'"2025-06-30"'},
            'cvi':{'grant_name':'ICJIA - CVI',
                    'grant_start':'"2024-10-01"',
                    'grant_end':'"2025-09-30"'},
            'r3':{'grant_name':'ICJIA - R3',
                    'grant_start':'"2024-11-01"',
                    'grant_end':'"2025-10-30"'},
            'scan':{'grant_name':'DFSS - SCAN',
                    'grant_start':'"2025-01-01"',
                    'grant_end':'"2025-12-31"'},
            'ryds':{'grant_name':'IDHS - RYDS',
                    'grant_start':'"2024-10-01"',
                    'grant_end':'"2025-09-30"'},
            'jac':{'grant_name': 'JAC - CVI',
                    'grant_start':'"2025-08-01"',
                    'grant_end':'"2027-07-31"'},
                    }


    def query_run(self, query):
        '''
        runs an SQL query (without a semicolon)
        format for a custom query: query_run(f"""query""")
        
        Parameters:
            query: SQL query
        '''
        query = text(f"{query};")
        if self.print_SQL is True:
            print(query)
        result = self.con.execute(query)
        data = result.fetchall()
        column_names = result.keys()
        df = pd.DataFrame(data, columns=column_names)
        if self.clipboard is True:
            df.to_clipboard(index=False)
        return df
    
    def query_modify(self, original_query, modification):
        '''
        edit a base SQL query, not particularly useful on its own

        Parameters:
            original_query: base SQL query
            modification: string to modify it
        '''
        # Use regular expressions to find the GROUP BY clause
        match = (re.compile(r"(?i)(\bGROUP\s+BY\b)",  re.IGNORECASE)).search(original_query)
        
        if match:
            # Insert the condition before the GROUP BY clause
            modified_query = original_query[:match.start()] + modification + ' ' + original_query[match.start():]
        else:
            # If no GROUP BY clause is found, just append the condition to the end of the query
            modified_query = original_query + ' ' + modification
        return modified_query
    
    def stints_cloud_run(self):
        '''
        abbreviated stints run for gcloud run
        '''
        if self.table != 'stints.active':
            stints_statements = f'''
            create table {self.table} as(
            select first_name, last_name, participant_id, program_type, program_start, program_end, 
            service_type, service_start, service_end, 
            case when (grant_start is null or grant_start <= {self.q_t2}) and (grant_end is null or grant_end >= {self.q_t1}) then grant_type else null end as grant_type,
            case when (grant_start is null or grant_start <= {self.q_t2}) and (grant_end is null or grant_end >= {self.q_t1}) then grant_start else null end as grant_start,
            case when (grant_start is null or grant_start <= {self.q_t2}) and (grant_end is null or grant_end >= {self.q_t1}) then grant_end else null end as grant_end,
            gender, race, age, birth_date, language_primary, case_managers, outreach_workers, attorneys,program_id, service_id from neon.big_psg
            join neon.basic_info using(participant_id)
            where ((program_start is null or program_start <= {self.q_t2}) and (program_end is null or program_end >= {self.q_t1})) and 
            (service_start is null or service_start <= {self.q_t2}) and (service_end is null or service_end >= {self.q_t1}));
            '''
            
            for statement in stints_statements.split(';'):
                if statement.strip():
                    self.con.execute(text(statement.strip() + ';'))

    def stints_run(self):
        '''
        runs the stints code
        '''
        stints_statements = f'''
drop table if exists stints.neon;
create table stints.neon as(
select first_name, last_name, participant_id, program_type, program_start, program_end, 
service_type, service_start, service_end, 
case when (grant_start is null or grant_start <= {self.q_t2}) and (grant_end is null or grant_end >= {self.q_t1}) then grant_type else null end as grant_type,
case when (grant_start is null or grant_start <= {self.q_t2}) and (grant_end is null or grant_end >= {self.q_t1}) then grant_start else null end as grant_start,
case when (grant_start is null or grant_start <= {self.q_t2}) and (grant_end is null or grant_end >= {self.q_t1}) then grant_end else null end as grant_end,
gender, race, age, birth_date, language_primary, case_managers, outreach_workers, attorneys,program_id, service_id from neon.big_psg
join neon.basic_info using(participant_id)
where ((program_start is null or program_start <= {self.q_t2}) and (program_end is null or program_end >= {self.q_t1})) and 
(service_start is null or service_start <= {self.q_t2}) and (service_end is null or service_end >= {self.q_t1}));

        drop table if exists stints.neon_chd;
        create table stints.neon_chd as(select * from stints.neon
        where program_type regexp 'chd|community navigation|violence prevention');

drop table if exists neon.client_teams;
create table neon.client_teams as
with active_services as(
select participant_id, first_name, last_name, service_type from {self.table}
where program_end is null and service_end is null),

all_staff as(
select participant_id, service_type, full_name as staff_name, team
from (select participant_id, full_name, 
case when a.staff_type like 'outreach%' then 'outreach'
when a.staff_type like 'case%' then 'case management'
when a.staff_type like '%attorney%' then 'legal'
end as 'service_type', team
from neon.assigned_staff a
left join neon.staff_ref r using(full_name)
where staff_end_date is null and staff_active='yes') e),

big_table as(
select * from active_services
left join all_staff using(participant_id, service_type)),

service_counts as (select participant_id, count(distinct service_type) as service_count
from big_table
where service_type not like 'therapy' and team is not null
group by participant_id),
team_counts as(select participant_id, team, count(team) as team_count from big_table
where team is not null
group by participant_id, team),

wide_teams as (select * from
(select distinct n.participant_id, n.first_name, n.last_name, case_managers, cm.team as cm_team, outreach_workers, ow.team as ow_team, attorneys, a.team as leg_team from 
(select * from 
{self.table} where program_end is null and service_end is null) n
left join neon.staff_ref cm on n.case_managers = cm.full_name
left join neon.staff_ref a on n.attorneys = a.full_name
left join neon.staff_ref ow on n.outreach_workers = ow.full_name) a),

better_teams as (
select participant_id, team as participant_team from (
select participant_id, count(distinct service_type) as team_service_count from big_table
where team is not null
group by participant_id
) s
join team_counts using(participant_id)
where team_service_count <= team_count)

select * from wide_teams
left join better_teams using(participant_id)
;

drop table if exists neon.cm_summary;
create table neon.cm_summary as(
with PARTS as 
(select first_name, last_name, participant_id, service_start as cm_start, case_managers as case_manager from stints.neon
where service_type = 'case management' and service_end is null),
ISP as (select participant_id, isp_start, latest_update as isp_update from neon.isp_tracker),
LINKS as(
select participant_id, max(linked_date) as latest_linkage, count(distinct linkage_id) as linkage_count
from
(select * from neon.linkages
join parts using(participant_id)
where (linked_date >=cm_start or start_date >= cm_start) and client_initiated = 'no') l
group by participant_id),
SESSIONS as(
select participant_id, max(case when successful_contact = "Yes" then session_date else null end) as latest_session, 
count(case when successful_contact = "Yes" then session_date else null end) as total_sessions, 
max(session_date) as latest_attempt, count(session_date) as total_attempts
from neon.case_sessions
join parts using(participant_id)
where session_date >= cm_start and contact_type = 'Case Management'
group by participant_id)

select * from parts
left join links using(participant_id)
left join isp using(participant_id)
left join sessions using(participant_id))
;

drop table if exists assess.outreach_tracker;
create table assess.outreach_tracker as(
with parts as (select distinct(participant_id), service_id, first_name, last_name, outreach_workers outreach_worker, service_start outreach_start from stints.active
where service_type like 'Outreach'),
eligibility as (select participant_id, count(distinct assessment_date) as elig_count, max(assessment_date) as latest_elig from assess.outreach_eligibility
group by participant_id),
assessment as (
select participant_id, count(distinct assessment_date) assessment_count, max(assessment_date) as latest_assessment from assess.safety_assessment
group by participant_id),
interv as(
select participant_id, count(distinct assessment_date) intervention_count, max(assessment_date) as latest_intervention
from assess.safety_intervention
group by participant_id),
intervention as(
select participant_id, intervention_count, latest_intervention, time_of_intervention from interv
join (select participant_id, time_of_assessment time_of_intervention, assessment_date latest_intervention
from assess.safety_intervention) i using(participant_id, latest_intervention))
select * from parts
left join eligibility using(participant_id)
left join assessment using(participant_id)
left join intervention using(participant_id)
order by outreach_start desc);

drop table if exists neon.bi_psg;

create table neon.bi_psg as(
with services as (
select program_id, service_id, grant_id, service_type, case when grant_type is null then "NO_GRANT" else grant_type end grant_type,
case when grant_start is null then service_start else grant_start end grant_start,
case when grant_end is null then service_end 
	when grant_end > service_end and service_end > grant_start then service_end
    else grant_end end grant_end
from neon.services
left join 
(select service_id, grant_id, grant_type, grant_start, grant_end
from neon.psg
where grant_end is null or grant_end > service_start) g using(service_id)),

merged as(
select participant_id, program_id, service_id, grant_id, program_type, case when service_type is null then "NO_SERVICE" else service_type end service_type, grant_type,
case when grant_start is null then program_start else grant_start end grant_start,
case when grant_end is null then program_end 
		when grant_end > program_end and program_end > grant_start then program_end
        else grant_end end grant_end
 from neon.programs
left join services using(program_id))

select participant_id, program_id, service_id, grant_id, program_type, service_type,
case when grant_type is null then "NO_SERVICE" else grant_type end grant_type,
grant_start start_date, grant_end end_date
from merged)
;

drop table if exists neon.service_staff;

create table neon.service_staff as (

with vs_ids as (
select assigned_staff_id, 'vs' from neon.assigned_staff
join (select participant_id, service_start, service_end from neon.services
where service_type = 'victim services') vs using(participant_id)
where full_name = 'Whitney Scott'),

staff_table as (
select participant_id, staff_name, staff_type,
case when staff_type = 'Case Manager' then "Case Management"
	when staff_type = 'Assigned Attorney' then "Legal"
    when staff_type = "Outreach Worker" then "Outreach"
    when staff_type = 'Therapist' then 'Therapy'
    when staff_type = 'Victim Services Advocate' then "Victim Services" end
    as service_type,
    staff_start_date staff_start,
    staff_end_date staff_end
from
(select participant_id, full_name staff_name, case when vs is not null then "Victim Services Advocate" else staff_type end staff_type,
staff_start_date, staff_end_date
from (select * from neon.assigned_staff
left join vs_ids using(assigned_staff_id)) v) vv),

abr_staff as(
select participant_id, service_id, staff_name, staff_start, staff_end from neon.services
left join staff_table using(participant_id, service_type)
where datediff(staff_start, service_start) > -100 and (service_end is null or (service_end >= staff_start)) and (staff_end is null or staff_end >= service_start)),

abr_w_rn as(select participant_id, service_id, staff_name, staff_start, staff_end, case when rn != 1 and staff_end is null then 1 else rn end rn
from (
select *, ROW_NUMBER() OVER (partition by service_id ORDER BY case when staff_end is null then 1 else 2 end, staff_end DESC) AS rn
from abr_staff) abr),

service_table as (
select service_id, group_concat(staff_name order by staff_start desc SEPARATOR ', ') assigned_staff, max(staff_start) staff_start, max(staff_end) staff_end
from 
(select * from abr_w_rn where rn = 1) ab
group by participant_id, service_id),

latest_service as (select participant_id, service_id, service_type, latest_service from neon.services
left join (select participant_id, service_type, max(service_start) service_start, 1 as latest_service from neon.services
group by participant_id, service_type) s using (participant_id, service_type, service_start))

select participant_id, service_id, service_type, case when assigned_staff is null then "MISSING" else assigned_staff end assigned_staff,
staff_start, staff_end, latest_service from
(select * from latest_service
left join service_table using(service_id)) s);

DROP TABLE IF EXISTS `stints`.`stint_count`;
create table stints.stint_count as(
WITH date_groups AS (
    SELECT 
        participant_id,
        program_id,
        program_start,
        program_end,
        CASE 
            WHEN LAG(program_end) OVER (PARTITION BY participant_id ORDER BY program_start) >= program_start 
            OR LAG(program_end) OVER (PARTITION BY participant_id ORDER BY program_start) IS NULL
            THEN 0
            ELSE 1
        END AS new_group
    FROM neon.programs
)
SELECT 
    participant_id, 
    1 + SUM(new_group) AS stint_count
FROM date_groups
GROUP BY participant_id);

DROP TABLE IF EXISTS `stints`.`stints_plus_stint_count`;
create table stints.stints_plus_stint_count as(
WITH DateGroups AS (
    SELECT 
        participant_id,
        program_start,
        program_end,
        CASE 
            WHEN LAG(program_end) OVER (PARTITION BY participant_id ORDER BY program_start) >= program_start THEN 0 
            ELSE 1 
        END AS is_new_group
    FROM neon.programs
),
Grouped AS (
    SELECT 
        participant_id,
        program_start,
        program_end,
        SUM(is_new_group) OVER (PARTITION BY participant_id ORDER BY program_start) AS group_num
    FROM DateGroups
)
SELECT participant_id, group_num stint_num, MIN(program_start) AS stint_start, MAX(program_end) AS stint_end
FROM Grouped
GROUP BY participant_id, stint_num
ORDER BY participant_id, stint_start);

        '''
        if self.mycase == True:
            stints_statements = stints_statements + ' ' + f'''drop table if exists neon.legal_mycase;
            create table neon.legal_mycase as(		with part as (select participant_id, program_start, program_end from neon.programs
            where program_type not like "RJCC|Violence Prevention"),

            stinted as( 
            select * from mycase.cases
            join (select distinct mycase_id, participant_id from part
            join mycase.cases using(participant_id)
            join stints.stint_count using(participant_id)
            where stint_count = 1 or 
            ((datediff(case_start, program_start) is null or datediff(case_start, program_start) >-60) 
            and (case_end is null or case_end > program_start) and (program_end is null or program_end > case_start))
            ) s using(participant_id, mycase_id)),
            
            base as (
            select participant_id, legal_id, mycase_id, sc.case_id, mycase_name, 
            case when case_start is null then arrest_date else case_start end as case_start, 
            case_end, case_status, case_type, violent, juvenile_adult, 
            attorney,
            case when class_prior_to_trial_plea like "fel%" then 'Felony'
            when class_prior_to_trial_plea like "mis%" then 'Misdemeanor'
            else 'Missing' end as class_type,
            class_prior_to_trial_plea, class_after_trial_plea, 
            case_outcome, case_outcome_date, sentence, probation_type, probation_requirements, sentence_length,
            expungable_sealable, expunge_seal_elig_date, expunge_seal_date, was_sentence_reduced, reduction_explain, probation_projected_end, probation_status_at_end, case_entered_date, outcome_entered_date,
            go_to_trial
            from stinted sc 
            left join neon.legal using(participant_id, legal_id)
            where (mycase_name not like "%traffic%" and sc.case_id not like '%tr%')),

            fel_rankings as (
            select distinct(mycase_id), fel_reduction from(
            select ranked.*, case when prior_rank < post_rank then 'reduced'
            when prior_rank > post_rank then 'increased'  
            when prior_rank = post_rank then 'remained'
            when prior_rank is not null and post_rank is null then 'missing post'
            else 'missing' end as fel_reduction
            from 
            (select mycase_id, class_prior_to_trial_plea, class_after_trial_plea, prior_rank ,ranking as post_rank from 
            (select mycase_id, class_prior_to_trial_plea, class_after_trial_plea, ranking as prior_rank from 
            (select distinct(mycase_id), class_prior_to_trial_plea, class_after_trial_plea from base) b
            left join misc.felony_classes f1 on b.class_prior_to_trial_plea = f1.felony) prior
            left join misc.felony_classes f2 on prior.class_after_trial_plea = f2.felony) ranked) reduced)
            
            select * from base
            join fel_rankings using(mycase_id));'''
        
        ### update teams




        for statement in stints_statements.split(';'):
            if statement.strip():
                self.con.execute(text(statement.strip() + ';'))

    def run_report(self, func_dict,*args, **kwargs):
        '''
        runs a desired report
        Parameters:
            func_dict: dictionary of functions to include, defaults to self.report_funcs. To use a different  
        '''       
        result_dict = {}
        for result_key, (func_name, func_args) in func_dict.items():
            func = getattr(self, func_name, None)
            if func and callable(func):
                try:
                    # Call the function with func_args and additional args/kwargs
                    result = func(*func_args, *args, **kwargs)
                    result_dict[result_key] = result
                except Exception as e:
                    result_dict[result_key] = f"Error: {str(e)}"

        return result_dict

    def run_projections(self, func_dict, *args, **kwargs):
        '''
        runs a report of grant projections
        function dictionary follows formt{grant_name:{func_dict}}
        '''
        def combine_initial_dictionaries():
            output_dict = {}
            for key in self.grant_dict.keys() & func_dict.keys():
                merged = dict(self.grant_dict[key])
                merged['queries'] = func_dict[key]
                output_dict[key] = merged
            return output_dict

        def create_outputs():
            for grant_name, grant_dict_vals in func_dict.items():
                grant_start = grant_dict_vals['grant_start']
                grant_end = grant_dict_vals['grant_end']
                queries = grant_dict_vals['queries']
                time_phases_dict = {'last_month':{
                    'table_name':f'grant_projections.{grant_name}_lmonth',
                    'phase_start': prev_first_str,
                    'phase_end': prev_last_str},
                'ytd':{
                    'table_name':f'grant_projections.{grant_name}_full',
                    'phase_start':grant_start,
                    'phase_end':grant_end}
                }
                outputs = {}
                for phase_name, phase_dict in time_phases_dict.items():
                    phase_table = phase_dict['table_name']
                    self.table_update(phase_table, True)
                    self.q_t1 = phase_dict['phase_start']
                    self.q_t2 = phase_dict['phase_end']
                    outputs[phase_name] = self.run_report(queries)
                func_dict[grant_name]['outputs'] = outputs

                ## projection part
                projection_reciprocal = percent_grant_complete(grant_start, grant_end)
                projected_outputs = {}

                for query_name, query_output in func_dict[grant_name]['outputs']['ytd'].items():
                    print(query_name, query_output)
                    proj_df = query_output.copy()
                    s = proj_df.select_dtypes(include=[np.number])*projection_reciprocal
                    s = s.round(0).astype(int)
                    proj_df[s.columns] = s
                    projected_outputs[query_name] = proj_df
                func_dict[grant_name]['outputs']['projections'] = projected_outputs


        
        def percent_grant_complete(grant_start, grant_end):
            # finish this 
            gstart = datetime.strptime(grant_start, '"%Y-%m-%d"').date()
            gend = datetime.strptime(grant_end, '"%Y-%m-%d"').date()
            pct_complete = ((prev_last - gstart).days / (gend - gstart).days)
            projection_reciprocal = (1 / pct_complete)
            return projection_reciprocal
        
        def format_outputs():
            clean_dict = {}
            for grant_name, output_dict in func_dict.items():
                formatted_outs = {}
                for query_name, query_df in output_dict['outputs']['ytd'].items():
                    non_numeric_cols = query_df.select_dtypes(exclude=[np.number]).columns.tolist()
                    numeric_cols = query_df.select_dtypes(include=[np.number]).columns.tolist()

                    # Build a dict of DataFrames for each period, indexed by non-numeric columns if present
                    if non_numeric_cols:
                        all_dfs = {
                            k: v[query_name].set_index(non_numeric_cols)[numeric_cols]
                            for k, v in output_dict['outputs'].items()
                        }
                        # Concat along columns, with period as first level
                        numeric_dfs = pd.concat(all_dfs, axis=1, names=["Period"])
                        # Combine non-numeric index and numeric MultiIndex columns
                        result = numeric_dfs
                    else:
                        all_dfs = {
                            k: v[query_name][numeric_cols]
                            for k, v in output_dict['outputs'].items()
                        }
                        numeric_dfs = pd.concat(all_dfs, axis=1, names=["Period"])
                        result = numeric_dfs

                    # Consistent int/float formatting
                    numeric_cols_result = result.select_dtypes(include=[np.number]).columns
                    numeric_data = result[numeric_cols_result].to_numpy()
                    has_decimals = np.any(~np.isclose(numeric_data, numeric_data.astype(int)))
                    if has_decimals:
                        result[numeric_cols_result] = result[numeric_cols_result].round(1)
                    else:
                        result[numeric_cols_result] = result[numeric_cols_result].astype('Int64')

                    formatted_outs[query_name] = result.fillna(0)
                clean_dict[grant_name] = formatted_outs
            return clean_dict



        prev_last = date.today().replace(day=1) - timedelta(days=1)
        prev_first = prev_last.replace(day=1)
        prev_last_str = f"'{prev_last.strftime('%Y-%m-%d')}'"
        prev_first_str = f"'{prev_first.strftime('%Y-%m-%d')}'"
        
        #doesn't actually matter much, 

        func_dict = combine_initial_dictionaries()
        create_outputs()
        cleaned_dict = format_outputs()
        return cleaned_dict

    
    def table_update(self, desired_table, update_default_table = False):
        if isinstance(desired_table, str):
            statements = None
            desired_table = desired_table.lower()
            if desired_table == 'rjcc':
                new_table = 'stints.rjcc'
                statements = f'''
                drop table if exists {new_table};
                create table {new_table} as(
                with mc as (
                select participant_id, mycase_id, mycase_name from mycase.case_stages
                where stage like "rjcc" and (stage_end is null or stage_end between {self.q_t1} and {self.q_t2}) and stage_start < {self.q_t2}),
                mcc as (select distinct mycase_id from mc
                join neon.legal_mycase using(participant_id, mycase_id, mycase_name)),

                mcunion as (
                select distinct mycase_id from (
                select distinct mycase_id from neon.legal_mycase
                where case_outcome like "%diverted%" or case_status like "%divers%"
                union all 
                select * from mcc) s),

                mccases as(
                select * from mcunion
                join neon.legal_mycase using(mycase_id)),
                mcparts as(
                select distinct(participant_id) from mccases)

                select * from mcparts
                join (select * from stints.neon
                join (select participant_id, count(distinct program_type) program_count
                from stints.neon
                group by participant_id) pc using(participant_id)) s using(participant_id)
                where program_type = 'rjcc' or program_count = 1);
                '''
            else:
                if desired_table in self.grant_dict: 
                    grant_type = f"'{self.grant_dict[desired_table]['grant_name']}'"
                    new_table = f'participants.{desired_table}'
                    statements = f'''
                    drop table if exists {new_table};
                    create table {new_table} as(
                    with idhs as (select *, case when (min_grant_start between {self.q_t1} and {self.q_t2}) then 'new' else 'continuing' end as new_client,
                    case when program_end between {self.q_t1} and {self.q_t2} then 'program'
                    when service_end between {self.q_t1} and {self.q_t2} or grant_end between {self.q_t1} and {self.q_t2} then 'service'
                    else null end as discharged_client
                    from {self.table}
                    join (select participant_id, min(grant_start) min_grant_start
                    from {self.table}
                    where grant_type = {grant_type}
                    group by participant_id) min using(participant_id)
                    where grant_type = {grant_type}),

                    discharged_prog as (select participant_id, service_count from (
                    select participant_id, count(participant_id) as service_count, count(case when discharged_client is not null then participant_id else null end) as discharge_ct from idhs
                    group by participant_id) s
                    where service_count = discharge_ct)

                    select participant_id, first_name, last_name, program_type, program_start, program_end, service_type, service_start, service_end, grant_type, grant_start, grant_end,
                    gender, race, age, birth_date, language_primary, case_managers, outreach_workers, attorneys, new_client, 
                    case when discharged_client = 'service' and service_count is not null then 'program' else discharged_client end as discharged_client from idhs
                    left join discharged_prog using(participant_id))
                    '''
                

            if statements:
                for statement in statements.split(';'):
                    if statement.strip():
                        self.con.execute(text(statement.strip() + ';'))
            if not statements:
                new_table = desired_table
            if update_default_table is True:
                self.table = new_table


class Audits(Tables):
    def __init__(self, t1, t2, engine, print_SQL = True, clipboard = False, default_table="stints.neon", mycase = True, cloud_run=False):
        super().__init__(t1, t2, engine, print_SQL, clipboard, default_table, mycase,cloud_run)

    @clipboard_decorator
    def program_lacks_services(self):
        '''
        Returns a table of active programs with no corresponding services.
        
        '''
        query = f'''
        with psg_table as (
        select * from (select first_name, last_name, participant_id from neon.basic_info) i
        join neon.big_psg using(participant_id)
        left join (select distinct participant_id, case_managers, outreach_workers, attorneys from {self.table}) sn using(participant_id)
        where program_type regexp 'chd|community navigation' and (program_end is null or program_end >= {self.q_t1}) and (service_end is null or service_end  >= {self.q_t1}))

        select * from psg_table where service_type is null
        '''
        df = self.query_run(query)
        return(df)

    @clipboard_decorator
    def service_lacks_grant(self):
        '''
        Returns a table of active services without a corresponding grant.
        '''
        query = f'''with psg_table as (
        select * from (select first_name, last_name, participant_id from neon.basic_info) i
        join neon.big_psg using(participant_id)
        where program_type = 'chd' and (program_end is null or  >= {self.q_t1}) and (service_end is null or service_end  >= {self.q_t1}))

        select * from psg_table where (service_type is not null and grant_type is null) or (service_end is null and grant_end is not null)
        order by participant_id asc'''
        df = self.query_run(query)
        return(df)
    
    @clipboard_decorator
    def service_lacks_staff(self, active_only = True):
        '''
        Returns a table of active services without a corresponding staff member.

        Parameters:
            active_only (Bool): only looks at clients with active programs/services
        '''

        if active_only:
            where_statement = f'''(program_end is null) and (service_end is null)'''

        else:
            where_statement = f'''(program_end is null or program_end >= {self.q_t1}) and (service_end is null or service_end >= {self.q_t1})'''

        query = f'''
        select participant_id, first_name, last_name, service_type, service_start from 
        (select distinct service_id, participant_id, first_name, last_name, service_type, service_start 
        from neon.big_psg
        join neon.basic_info using(participant_id)
        join (
        select service_id, assigned_staff from neon.service_staff
        where latest_service = 1) s using(service_id)
        where assigned_staff = 'missing' and {where_statement}) b
        order by service_type, last_name
        '''
        df = self.query_run(query)
        return(df)

    @clipboard_decorator
    def staff_lacks_service(self):
        '''
        Returns a table of staff members assigned to clients without a corresponding service type
        '''

        query = f'''with serv as (
        select participant_id, program_type, service_type, service_start, service_end from neon.big_psg
        where service_end is null and program_end is null and service_type is not null),

        concat_serv as (
        select participant_id, group_concat(distinct service_type) as active_services
        from serv
        group by participant_id
        ),

        staf as(
        select participant_id, first_name, last_name, active_services,full_name as staff_name, staff_type, staff_start_date, case 
            when staff_type = 'case manager' then 'Case Management'
            when staff_type = 'assigned attorney' then 'Legal'
            else 'Outreach' end as 'service_type'  from neon.assigned_staff
        join (select first_name, last_name, participant_id from neon.basic_info) n using(participant_id)
        join concat_serv using(participant_id)
        join (select distinct participant_id from serv) s using(participant_id)
        where staff_active = 'Yes')

        select * from staf
        left join serv using(participant_id, service_type)
        where program_type is null;
        '''
        df = self.query_run(query)
        return(df)

    def legal_audit_lawyers(self, func_dict = None):
        '''
        Returns several joined tables of missing legal data

        Parameters:
            func_dict: a dictionary of functions to use

        Note:
            Audit - Legal Data
        '''


        def caseid_NAN(active_only = True, new_cases = False):
            active_only = f"and participant_id in (select distinct participant_id from {self.table} where service_type = 'legal' and program_end is null and service_end is null)" if active_only else ''
            new_cases = f'and case_start between {self.q_t1} and {self.q_t2}' if new_cases else f'and ((case_outcome_date is null and (case_end is null or case_end > {self.q_t1})) or case_outcome_date between {self.q_t1} and {self.q_t2})'

            query = f'''
            with base as (
            select * from neon.legal_mycase
            where mycase_id not regexp "31580789"
            {active_only}
            {new_cases})
            select attorney, "CaseID missing from MyCase" as item,case_id,mycase_name, participant_id from base
            where case_id = "NAN"'''

            df = self.query_run(query)
            return(df)

        def custody_status(active_only = True, new_cases = False):
            active_only = f"and participant_id in (select distinct participant_id from {self.table} where service_type = 'legal' and program_end is null and service_end is null)" if active_only else ''
            new_cases = f'and case_start between {self.q_t1} and {self.q_t2}' if new_cases else f'and ((case_outcome_date is null and (case_end is null or case_end > {self.q_t1})) or case_outcome_date between {self.q_t1} and {self.q_t2})'

            query = f'''
            with base as (
            select * from neon.legal_mycase
            where mycase_id not regexp "31580789"
            {active_only}
            {new_cases}),

                custody_status as(
            select participant_id, custody_status, custody_status_date from neon.custody_status
            join 
            (select participant_id, max(custody_status_id) as custody_status_id from neon.custody_status
            group by participant_id) ncs using(participant_id, custody_status_id)),

            status_table as(
            select mycase_name, attorney, participant_id, case_start, case_status, case_outcome, sentence, case_outcome_date, custody_status, custody_status_date, client_name from base
            left join custody_status using(participant_id)
            left join (select participant_id, concat(first_name, " ", last_name) client_name from neon.basic_info) b using(participant_id)),
            
            missing as(select attorney, "Client lacks custody status" as item, participant_id, client_name, null as explanation from
            (select distinct client_name, attorney, min(case_start) case_start, participant_id
            from status_table
            where custody_status_date is null
            group by client_name, attorney, participant_id) m),
            
            needs_updating as(
            select attorney, item, participant_id, client_name, concat("Status: ", custody_status, ". Latest outcome:", case_outcome) explanation
            from 
            (select distinct(participant_id) participant_id, attorney, "Custody status needs updating" as item, client_name, 
            concat(case when sentence is null then case_outcome else sentence end, " (", case_outcome_date,")") case_outcome, 
            concat(custody_status, " (", custody_status_date,")") custody_status
            from status_table
            join (select participant_id, max(case_outcome_date) case_outcome_date
            from status_table
            group by participant_id) s using(participant_id, case_outcome_date) 
            where case_outcome_date > custody_status_date and(sentence like 'IDOC/IDJJ' and custody_status not like 'in custody') or 
            (custody_status = 'in custody' and (sentence not like 'IDOC/IDJJ' or sentence is null)))n)
            
            select * from needs_updating
            union all
            select * from missing
            order by item desc, attorney asc'''

            df = self.query_run(query)
            return(df)

        def misaligned_attorneys(active_only = True, new_cases = False):
            active_only = f"and participant_id in (select distinct participant_id from {self.table} where service_type = 'legal' and program_end is null and service_end is null)" if active_only else ''
            new_cases = f'and case_start between {self.q_t1} and {self.q_t2}' if new_cases else f'and ((case_outcome_date is null and (case_end is null or case_end > {self.q_t1})) or case_outcome_date between {self.q_t1} and {self.q_t2})'

            query = f'''
            with neon_staff as(select participant_id, assigned_staff as neon_attorney from 
            (select distinct participant_id, service_id from stints.neon_chd where service_type = 'legal' and program_end is null and service_end is null) l
            join neon.service_staff using(participant_id, service_id)),
            
            base as (
            select * from neon.legal_mycase
            where mycase_id not regexp "31580789"
            {active_only}
            {new_cases}),
            

            messy as(
            select case when neon_attorney like "MISSING" then "Attorney not in Neon"  when attorney like "MISSING" then "Attorney not in MyCase" else "MyCase attorney different from Neon" end as 'item', mycase_name, participant_id, case_id, attorney as mycase_attorney, neon_attorney, 
            case when attorney like "Thomas" then neon_attorney else attorney end as attorney, concat("MyCase: ", attorney, ". Neon: ", neon_attorney) as explanation
            from 
            (select case_id, mycase_name, case_start, mycase_id, participant_id, attorney, SUBSTRING_INDEX(TRIM(attorney), ' ', -1) as l_name, neon_attorney from base
            left join neon_staff using(participant_id)) atty
            where INSTR(COALESCE(neon_attorney, ''), COALESCE(l_name, '')) = 0)
            
            select attorney, item, mycase_name, participant_id, explanation
            from messy 
            '''
            df = self.query_run(query)
            return(df)

        def missing_from_neon(active_only = True, new_cases = False):
            active_only = f"and participant_id in (select distinct participant_id from {self.table} where service_type = 'legal' and program_end is null and service_end is null)" if active_only else ''
            new_cases = f'and case_start between {self.q_t1} and {self.q_t2}' if new_cases else f'and ((case_outcome_date is null and (case_end is null or case_end > {self.q_t1})) or case_outcome_date between {self.q_t1} and {self.q_t2})'
            
            query = f'''
            select attorney, 'Case missing from NeonOne' as item, case_id, mycase_name, participant_id from neon.legal_mycase
            where legal_id is null and mycase_id not regexp "31580789"
            {active_only}
            {new_cases}
            '''
            df = self.query_run(query)
            return(df)
        
        def missing_felony_class(active_only = True, new_cases = False):
            active_only = f"and participant_id in (select distinct participant_id from {self.table} where service_type = 'legal' and program_end is null and service_end is null)" if active_only else ''
            new_cases = f'and case_start between {self.q_t1} and {self.q_t2}' if new_cases else f'and ((case_outcome_date is null and (case_end is null or case_end > {self.q_t1})) or case_outcome_date between {self.q_t1} and {self.q_t2})'

            query = f'''
            with base as (
            select * from neon.legal_mycase
            where mycase_id not regexp "31580789"
            {active_only}
            {new_cases}),
            
            ongoing_cases as(
            select attorney, "Case missing highest felony class" as item, case_id, mycase_name, participant_id from base
            where legal_id is not null and class_prior_to_trial_plea is null),

            closed_cases as(
            select attorney, "Case missing post-trial highest felony class" as item, case_id, mycase_name, participant_id from base
            where case_status regexp "Case closed|Post-sentencing|Probation" and class_after_trial_plea is null)

            select * from ongoing_cases
            union all
            select * from closed_cases
            '''
            df = self.query_run(query)
            return(df)

        def undated_outcome(active_only = True, new_cases = False):
            active_only = f"and participant_id in (select distinct participant_id from {self.table} where service_type = 'legal' and program_end is null and service_end is null)" if active_only else ''
            new_cases = f'and case_start between {self.q_t1} and {self.q_t2}' if new_cases else f'and ((case_outcome_date is null and (case_end is null or case_end > {self.q_t1})) or case_outcome_date between {self.q_t1} and {self.q_t2})'

            query = f'''
            with base as (
            select * from neon.legal_mycase
            where mycase_id not regexp "31580789"
            {active_only}
            {new_cases})

            select attorney, participant_id, mycase_name, "Case outcome missing date" as item from base
            where case_outcome is not null and case_outcome_date is null
            '''

            df = self.query_run(query)
            return(df)

        if not func_dict:
            func_dict = {
            'missing_from_neon': (missing_from_neon, (True,False,)),
            'missing_felony_class': (missing_felony_class, (True,False,)),
            'caseid_NAN': (caseid_NAN, (True,False,)),
            'misaligned_attorneys': (misaligned_attorneys, (True,False,)),
            'custody_status': (custody_status, (True,False,)),
            'undated_outcome': (undated_outcome, (True,False,)),
        }
        
        result_dict = {}
        for result_key, (func, func_args) in func_dict.items():
            try:
                result_dict[result_key] = func(*func_args)
            except Exception as e:
                print(result_key, f"Error: {str(e)}")
        
        result_df = pd.concat(result_dict.values(), ignore_index=True)
        result_df['mycase_name'] = result_df['mycase_name'].fillna(result_df['client_name'])
        result_df = result_df.rename(columns={'mycase_name':'case/client name'})
        result_df = result_df[['attorney','participant_id','case/client name', 'item', 'explanation']].sort_values(by=['attorney','participant_id'])
        
        return result_df

    ### SERVICE-SPECIFIC SHINY STUFF
    @clipboard_decorator
    def active_session_tally(self, service_type = 'Case Management', successful_only = True):
        '''
        Counts the number of active sessions for each participant_id

        Parameters:
            service_type: what service to count sessions of. Defaults to 'Case Management'
            successful_only: Whether to only include successful contacts. Defaults to True
        
        Note:
            Case Sessions - Number of Case Sessions per Client
        '''

        successful_statement = "and successful_contact = 'yes'" if successful_only else ''
        query = f'''
        select * from
        (select distinct participant_id from stints.active where service_type = '{service_type}') st 
        left join (select participant_id, count(participant_id) total_sessions, max(session_date) latest_session from neon.case_sessions
        join (select participant_id, service_start from stints.active where service_type = '{service_type}') st using(participant_id)
        where contact_type = '{service_type}' and session_date >= service_start {successful_statement}
        group by participant_id) al using(participant_id)
        '''
        df = self.query_run(query)
        return(df)

    @clipboard_decorator
    def cm_isp_status(self):
        '''
        Table of ISP status for active clients

        Note:
            ISPs - Statuses for Active Clients
        '''
        query = f'''
        select participant_id, latest_update, total_goals, goals_completed, 
            case when latest_update is null then 'Needs Initial'
        when datediff(DATE_ADD(latest_update, INTERVAL 3 MONTH), CURDATE()) <0 then 'Needs Update'
        else 'Up to Date' end 'ISP_Status'
        from neon.isp_tracker
        right join (select distinct participant_id from stints.active) a using(participant_id)
        '''
        df = self.query_run(query)
        return(df)
    
    @clipboard_decorator
    def cm_linkage_totals(self):
        '''
        Table of linkage totals for active clients

        Note:
            Linkages - Totals for Active Clients
        '''
        query = f'''
        select * from (select distinct participant_id from stints.active where service_type = 'case management') stint
        left join(
        select participant_id, count(distinct linkage_id) total_linkages, max(linked_date) as latest_linkage from neon.linkages
        join (select * from stints.active where service_type = 'case management') s using(participant_id)
        where service_start <= linked_date and client_initiated = 'no'
        group by participant_id) p using(participant_id)
        '''
        df = self.query_run(query)
        return(df)
        
    @clipboard_decorator
    def outreach_missing_assessments(self):
        '''
        Returns a table of the count and names of missing outreach assessments for clients. 

        Note:
            Outreach - Missing Assessment Information
        '''
        query = f'''
        with parts as (select distinct participant_id, service_start from stints.active where service_type = 'outreach'),
		eligibility as (select participant_id, count(distinct assessment_date) as elig_count, max(assessment_date) as latest_elig from assess.outreach_eligibility
        group by participant_id),
    	assessment as (
        select participant_id, count(distinct assessment_date) assessment_count, max(assessment_date) as latest_assessment from assess.safety_assessment
        group by participant_id),
      interv as(
        select participant_id, count(distinct assessment_date) intervention_count, max(assessment_date) as latest_intervention
        from assess.safety_intervention
        group by participant_id),
     intervention as(
        select participant_id, intervention_count, latest_intervention, time_of_intervention from interv
        join (select participant_id, time_of_assessment time_of_intervention, assessment_date latest_intervention
        from assess.safety_intervention) i using(participant_id, latest_intervention)),
    big_table as (
      select * from parts
        left join eligibility using(participant_id)
        left join assessment using(participant_id)
        left join intervention using(participant_id)),
    str_table as(
		select participant_id,
    case when elig_count is null then "Eligibility Screening" else null end elig_count,
    case when assessment_count is null then "Safety Assessment" else null end assess_count,
		case when intervention_count is null then "Initial Safety Intervention"
     	when time_of_intervention like 'Intake' and datediff(CURDATE(), latest_intervention) > 100 then "Updated Safety Intervention" else null end intervention_count
    from big_table)
    select participant_id, 
    ((elig_count IS NOT NULL) +
    (assess_count IS NOT NULL) +
    (intervention_count IS NOT NULL) ) num_missing_assessments,
    CONCAT_WS( ', ', elig_count, assess_count, intervention_count) missing_assessments
    from str_table
        '''
        df = self.query_run(query)
        return(df)


class Queries(Audits):
    def __init__(self, t1, t2, engine, print_SQL = True, clipboard = False, default_table="stints.neon", mycase = True,cloud_run=False):
        super().__init__(t1, t2, engine, print_SQL, clipboard, default_table, mycase,cloud_run)
    '''
    Sets up a table for a given timeframe

    Parameters:
        t1: start date, formatted as "YYYY-MM-DD"
        t2: end date, formatted as "YYYY-MM-DD"
        print_sql (Bool): whether to print the SQL statements when run, defaults to True
        clipboard (Bool): whether to copy the output table to your clipboard, defaults to False
        default_table: the source table to run queries on. defaults to "stints.neon", can also use "stints.neon_chd", or a participants table
        mycase(Bool): whether the user has a formatted MyCase SQL database, defaults to True
        establishes settings and runs stints for the desired time period

    Examples:
        Create an object of all clients in Neon in 2024 ::
        
            e = Queries(t1 = '2024-01-01', t2='2024-12-31')
        
        Create an object of CHD clients in Q3 of 2024::

            e = Queries(t1= '2024-07-01', t2= '2024-09-30', default_table = stints.neon_chd)
    '''

    @clipboard_decorator
    def assess_assm(self, cutoff_score = 2, score_date = 'min'):
        '''
        Counts clients with ASSM scores in a certain range.

        Parameters:
            cutoff_score: the upper bound of score to include. Defaults to 2
            score_date (str): 'min' returns the earliest score, 'max' returns the latest. Defaults to 'min'

        Examples:
            Get a count of clients with their earliest ASSM scores between 1-2::

                e.assess_assm()

            Get a count of clients with their latest ASSM scores between 1-3::
                
                e.assess_assm(cutoff_score=3, score_date='max')
        
        Note:
            Assessments - ASSM Scores by Category

        '''

        query = f'''
        with elig_assess as(
        select * from assess.assm
        join (select distinct participant_id, program_start from {self.table}
        where program_type regexp ".*CHD.*|.*Navigation.*") n using(participant_id)
        where assessment_date >= program_start),

        main_assm as(select * from elig_assess
        join (select distinct participant_id, {score_date}(assessment_date) assessment_date from elig_assess group by participant_id) e using (participant_id, assessment_date))

        select 'crisis_scores' as domain,
        count(distinct case when edu_adult between 1 and {cutoff_score} or edu_juv between 1 and {cutoff_score} then participant_id else null end) as edu,
        count(distinct case when employment between 1 and {cutoff_score} then participant_id else null end) as employment,
        count(distinct case when family_relations between 1 and {cutoff_score} then participant_id else null end) as family_relations,
        count(distinct case when peer_relations between 1 and {cutoff_score} then participant_id else null end) as peer_relations,
        count(distinct case when legal between 1 and {cutoff_score} then participant_id else null end) as legal,
        count(distinct case when community_involvement between 1 and {cutoff_score} then participant_id else null end) as community_involvement,
        count(distinct case when mental_health between 1 and {cutoff_score} then participant_id else null end) as mental_health,
        count(distinct case when substance_abuse between 1 and {cutoff_score} then participant_id else null end) as substance_abuse,
        count(distinct case when safety between 1 and {cutoff_score} then participant_id else null end) as safety,
        count(distinct case when housing between 1 and {cutoff_score} then participant_id else null end) as housing,
        count(distinct case when food between 1 and {cutoff_score} then participant_id else null end) as food,
        count(distinct case when life_skills between 1 and {cutoff_score} then participant_id else null end) as life_skills,
        count(distinct case when mobility between 1 and {cutoff_score} then participant_id else null end) as mobility
        from main_assm
        UNION ALL
        select 'total_participants' as domain,
        count(distinct case when edu_adult is not null or edu_juv is not null then participant_id else null end) as edu,
        count(distinct case when employment is not null then participant_id else null end) as employment,
        count(distinct case when family_relations is not null then participant_id else null end) as family_relations,
        count(distinct case when peer_relations is not null then participant_id else null end) as peer_relations,
        count(distinct case when legal is not null then participant_id else null end) as legal,
        count(distinct case when community_involvement is not null then participant_id else null end) as community_involvement,
        count(distinct case when mental_health is not null then participant_id else null end) as mental_health,
        count(distinct case when substance_abuse is not null then participant_id else null end) as substance_abuse,
        count(distinct case when safety is not null then participant_id else null end) as safety,
        count(distinct case when housing is not null then participant_id else null end) as housing,
        count(distinct case when food is not null then participant_id else null end) as food,
        count(distinct case when life_skills is not null then participant_id else null end) as life_skills,
        count(distinct case when mobility is not null then participant_id else null end) as mobility
        from main_assm;
        '''
        df = self.query_run(query)
        df = df.set_index('domain').T
        print(df)
        df['crisis_percentage'] = df['crisis_scores']/df['total_participants']
        df = df.reset_index().sort_values(by=['crisis_percentage'], ascending=False)
        df['crisis_percentage'] = df['crisis_percentage'].apply(lambda x: "{:.2f}%".format(x * 100))
        return(df)
    
    def assess_assm_improvement(self, timeframe = False, isp_goals = False):
        '''
        Returns the number of clients with ASSM improvements in each category


        Parameters:
            timeframe (Bool): whether to only look at ASSM scores in the timeframe (and a client's original). Defaults to False
            isp_goals (Bool): Only look at score improvements for ISP goal domains. Defaults to False

        Example:
            Get a breakdown of ASSM score improvements in the timeframe::
                
                e.assess_assm_improvement(timeframe = True)
            
            Get the number of clients whose ASSM increased in an ISP goal area::

                e.assm_improvement(isp_goals = True)

        Note:
            Assessments - ASSM Score Changes
        '''
        
        where_statement = f'''where min_date = assessment_date or assessment_date between {self.q_t1} and {self.q_t2}''' if timeframe else ''
        
        
        query = f'''
            with parts as (
                select distinct participant_id, min(program_start) program_start
                from {self.table}
                where program_type not like "RJCC"
                group by participant_id),
            full_assm as (
            select * from assess.assm
            join parts using(participant_id)
            where assessment_date >= program_start and sum > 0),

            earliest as (
            select * from full_assm
            join 
                (select participant_id, min(assessment_date) assessment_date
                from full_assm
                group by participant_id) a using(participant_id, assessment_date)),

            base as(select * from full_assm
            join (select participant_id, assessment_date min_date from earliest) a using(participant_id)
            {where_statement}
            ),

            max as (
            select participant_id, count(participant_id) num_assms, max(edu_adult) max_edu_adult,
            max(edu_juv) max_edu_juv, max(employment) max_employment, max(family_relations) max_family_relations,
            max(peer_relations) max_peer_relations, max(legal) max_legal, max(community_involvement) max_community_involvement,
            max(mental_health) max_mental_health, max(substance_abuse) max_substance_abuse, max(safety) max_safety, max(housing) max_housing,
            max(food) max_food, max(life_skills) max_life_skills, max(mobility) max_mobility, max(sum) max_sum
            from base
            group by participant_id
            ),
            matrix as (
            SELECT 
                m.participant_id,
                m.num_assms,
                CASE WHEN m.max_edu_adult > b.edu_adult THEN 1 ELSE 0 END AS edu_adult,
                CASE WHEN m.max_edu_juv > b.edu_juv THEN 1 ELSE 0 END AS edu_juv,
                CASE WHEN m.max_employment > b.employment THEN 1 ELSE 0 END AS employment,
                CASE WHEN m.max_family_relations > b.family_relations THEN 1 ELSE 0 END AS family_relations,
                CASE WHEN m.max_peer_relations > b.peer_relations THEN 1 ELSE 0 END AS peer_relations,
                CASE WHEN m.max_legal > b.legal THEN 1 ELSE 0 END AS legal,
                CASE WHEN m.max_community_involvement > b.community_involvement THEN 1 ELSE 0 END AS community_involvement,
                CASE WHEN m.max_mental_health > b.mental_health THEN 1 ELSE 0 END AS mental_health,
                CASE WHEN m.max_substance_abuse > b.substance_abuse THEN 1 ELSE 0 END AS substance_abuse,
                CASE WHEN m.max_safety > b.safety THEN 1 ELSE 0 END AS safety,
                CASE WHEN m.max_housing > b.housing THEN 1 ELSE 0 END AS housing,
                CASE WHEN m.max_food > b.food THEN 1 ELSE 0 END AS food,
                CASE WHEN m.max_life_skills > b.life_skills THEN 1 ELSE 0 END AS life_skills,
                CASE WHEN m.max_mobility > b.mobility THEN 1 ELSE 0 END AS mobility,
                CASE WHEN m.max_sum > b.sum THEN 1 ELSE 0 END AS sum
            FROM 
                max m
            JOIN 
                earliest b ON m.participant_id = b.participant_id),
            
            longie as(
            select participant_id, 'SUM' as assm_domain
            from matrix where sum = 1
            union all
            select participant_id, 'Education (Adult)' as goal_domain
            from matrix where edu_adult = 1
            union all
            select participant_id, 'Education (Juvenile)' as goal_domain
            from matrix where edu_juv = 1
            union all
            select participant_id, 'Employment' as goal_domain
            from matrix where employment = 1
            union all
            select participant_id, 'Family Relations' as goal_domain
            from matrix where family_relations = 1
            union all
            select participant_id, 'Peer Relations' as goal_domain
            from matrix where peer_relations = 1
            union all
            select participant_id, 'Legal' as goal_domain
            from matrix where legal = 1
            union all
            select participant_id, 'Community Involvement' as goal_domain
            from matrix where community_involvement = 1
            union all
            select participant_id, 'Mental Health' as goal_domain
            from matrix where mental_health = 1
            union all
            select participant_id, 'Substance Abuse' as goal_domain
            from matrix where substance_abuse = 1
            union all
            select participant_id, 'Safety' as goal_domain
            from matrix where safety = 1
            union all
            select participant_id, 'Housing' as goal_domain
            from matrix where housing = 1
            union all
            select participant_id, 'Food' as goal_domain
            from matrix where food = 1
            union all
            select participant_id, 'Life Skills' as goal_domain
            from matrix where life_skills = 1
            union all
            select participant_id, 'Mobility' as goal_domain
            from matrix where Mobility = 1)
        '''
        if isp_goals:
            addendum = f'''
            ,
            participant_goals as (
            select distinct participant_id, num_assms, goal_domain from neon.isp_goals
            join (select participant_id, isp_id from neon.isp_goals
                join (select participant_id, program_start from parts) p using(participant_id)
                where isp_start >= program_start) i using(participant_id, isp_id)
            join (select participant_id, num_assms from max) m using(participant_id)
            where num_assms > 1),

            domain_merge as (
            select g.participant_id, goal_domain, assm_domain from participant_goals g
            left join longie l on g.participant_id = l.participant_id and g.goal_domain = l.assm_domain)

            select 'ANY DOMAIN' as domain, count(distinct participant_id) total_participants, count(distinct case when assm_domain is not null then participant_id else null end) improved_score from domain_merge
            join participant_goals using(participant_id)
            union all
            select goal_domain, count(participant_id) total_participants, count(case when assm_domain is not null then 1 else null end) improved_score
            from domain_merge
            group by goal_domain
            order by total_participants desc
            '''
        else:
            addendum = f''' 
            select 'Have an ASSM' assm_domain, count(distinct participant_id) total_clients from max
            union all
            select 'Multiple ASSMs' assm_domain, count(distinct participant_id) total_clients from max
            where num_assms > 1
            union all
            select assm_domain, count(distinct participant_id)
            from longie
            group by assm_domain
            order by case when assm_domain = 'Have an ASSM' then 1 
            when assm_domain = 'Multiple ASSMs' then 2 
            when assm_domain = 'sum' then 3
            else 4
            end, total_clients desc'''
        
        query = query + ' ' + addendum
        df = self.query_run(query)
        return(df)

    @clipboard_decorator
    def assess_outreach_completion(self):
        '''
        Returns a table of outreach assessments (and total number of assessments) and their completion rates within 30 days/while working with a client/
        
        Note:
            Assessments - Outreach Completion Rates
        '''

        query = f'''
        with parts as (select distinct(participant_id), first_name, last_name, outreach_workers outreach_worker, service_start outreach_start from {self.table}
        where service_type like 'Outreach'),
        eligibility as (select participant_id, count(distinct assessment_date) as elig_count, max(assessment_date) as latest_elig, min(abs(datediff(outreach_start, assessment_date))) time_between, 'Eligibility' assess_type
        from assess.outreach_eligibility
        join parts using(participant_id)
        group by participant_id),
        assessment as (
        select participant_id, count(distinct assessment_date) assessment_count, max(assessment_date) as latest_assessment, min(abs(datediff(outreach_start, assessment_date))) time_between, 'Safety Assessment' from assess.safety_assessment
        join parts using(participant_id)
        group by participant_id),
        interv as(
        select participant_id, count(distinct assessment_date) intervention_count, max(assessment_date) as latest_intervention, min(abs(datediff(outreach_start, assessment_date))) time_between
        from assess.safety_intervention
        join parts using(participant_id)
        group by participant_id),
        intervention as(
        select participant_id, intervention_count, latest_intervention, time_between, 'Safety Intervention' from interv
        join parts using(participant_id)
        join (select participant_id, time_of_assessment time_of_intervention, assessment_date latest_intervention
        from assess.safety_intervention) i using(participant_id, latest_intervention)),

        longie as(select * from parts
        join
        (select * from eligibility
        union all
        select * from assessment
        union all
        select * from intervention)l using(participant_id)),

        by_assessment as (select assess_type, count(distinct participant_id) total_assessed,
        count(distinct case when time_between <= 31 then participant_id end) assessed_in_30
        from longie
        group by assess_type),

        long_and_unioned as
        (select * from by_assessment
        union all
        select 
        concat('Total: ', total_assessed) num_assesses,
        count(distinct participant_id) num_assessed, 
        count(distinct case when assessed_in_30 = total_assessed then participant_id else null end) assessed_in_30 
        from
        (select participant_id, count(distinct assess_type) total_assessed,
        count(distinct case when time_between <= 31 then assess_type end) assessed_in_30
        from longie
        group by participant_id)a
        group by total_assessed
        union all
        select 'Total: Any Assessment',
        count(distinct participant_id) num_assessed, 
        count(distinct case when assessed_in_30 = total_assessed then participant_id else null end) assessed_in_30
        from (select participant_id, count(distinct assess_type) total_assessed,
        count(distinct case when time_between <= 31 then assess_type end) assessed_in_30
        from longie
        group by participant_id)a)

        select assess_type, 
        CONCAT(ROUND((total_assessed / total_clients) * 100, 0), '%') AS pct_assessed,
        CONCAT(ROUND((assessed_in_30 / total_assessed) * 100, 0), '%') AS pct_assessed_in_30,
        CONCAT(ROUND((assessed_in_30 / total_clients) * 100, 0), '%') AS pct_of_total_assessed_in_30
        from
        (select assess_type, 
        (select count(distinct participant_id) from parts) as total_clients, 
        total_assessed, assessed_in_30 
        from long_and_unioned)b

        '''
        df = self.query_run(query)
        return(df)

    @clipboard_decorator
    def assess_cm_tracker(self, active_only = True, new_clients = False, summary_table = False):
        '''
        Returns a table of clients with CM assessments and their latest assessment dates.
        
        Parameters:
            active_only (Bool): whether to only look at clients with active CM services. Defaults to True
            new_clients_only (Bool): whether to only look at clients with new CM services in the timeframe. Defaults to False
            summary_table (Bool): whether to return a summary table of clients with assessments. Defaults to False
            
        Note:
            Assessments - Case Management Tracker    
        '''
        
        new_clients_str = ''
        if new_clients:
            if self.table.startswith('stints.'):
                new_clients_str = f'b.program_start between {self.q_t1} and {self.q_t2}' if active_only else ''
            else:
                new_clients_str = 'new_client = "new"'
        
        active_clients = "b.case_managers is not null" if not active_only else ''
        better_parameters = 'where ' + ' and '.join(filter(None, [active_clients, new_clients_str])) if any([active_clients, new_clients_str]) else ''

        query = f'''
        with all_parts as (select distinct(participant_id), b.first_name, b.last_name, b.case_managers case_manager from neon.basic_info b
        join {self.table} using(participant_id)
        {better_parameters}),
        cm_assesses as (
        select * from all_parts
        left join (select participant_id, latest_update isp_latest from neon.isp_tracker) isp using(participant_id)
        left join 
        (select participant_id, max(assessment_date) needs_latest from assess.needs_assessment
        group by participant_id) needs using(participant_id)
        left join (
        select participant_id, latest_date assm_latest from assess.cm_tracker
        where assessment_type = 'assm') assm using (participant_id)
        left join (
        select participant_id, latest_date bp_latest from assess.cm_tracker
        where assessment_type = 'bp') bp using (participant_id)
        left join (
        select participant_id, latest_date cdc_latest from assess.cm_tracker
        where assessment_type = 'cdc') cdc using (participant_id)
        left join (
        select participant_id, latest_date fcs_latest from assess.cm_tracker
        where assessment_type = 'fcs') fcs using (participant_id)
        left join (
        select participant_id, latest_date pcl_latest from assess.cm_tracker
        where assessment_type = 'pcl') pcl using (participant_id))
        '''
        if summary_table: 
            addendum = f'''
    select count(distinct participant_id) 'total',
        count(distinct case when isp_latest is not null then participant_id else null end) 'ISP',
        count(distinct case when needs_latest is not null then participant_id else null end) 'Needs_Assess',
        count(distinct case when assm_latest is not null then participant_id else null end) 'ASSM',
        count(distinct case when ((bp_latest is not null) + (cdc_latest is not null) + (fcs_latest is not null) + (pcl_latest is not null)) = 4 then participant_id else null end) '4_assesses',
        count(distinct case when ((bp_latest is not null) + (cdc_latest is not null) + (fcs_latest is not null) + (pcl_latest is not null)) >= 2 then participant_id else null end) '2plus_assesses',
        count(distinct case when bp_latest is not null then participant_id else null end) 'BP',
        count(distinct case when cdc_latest is not null then participant_id else null end) 'CDC',
        count(distinct case when fcs_latest is not null then participant_id else null end) 'FCS',
        count(distinct case when pcl_latest is not null then participant_id else null end) 'PCL'
        from cm_assesses
            '''
        else:
            addendum = f'''select * from cm_assesses'''

        query = query + ' ' + addendum
        df = self.query_run(query)
        if summary_table:
            df = df.T.rename(columns={0:'Total Clients'})
            total = df.loc['total', 'Total Clients']
            df['Percentage'] = df['Total Clients'] / total * 100
            df = df.round(1).astype({'Percentage': 'float64'}).fillna(0)
        return(df)

    @clipboard_decorator
    def assess_score_change(self, timeframe=True, min_score = None):
        '''
        Returns a table of CM assessments & their score changes.

        Parameters:
            timeframe (Bool): only looks at score changes in the timeframe. Defaults to True
            min_score: the lowest pre-assessment score to consider. Defaults to None 
        
        Examples:
            Get count of assessment score changes within timeframe::
                
                e.assess_score_change()
            
            Get all-time count of assessment score changes for initial scores of 32+::
                
                e.assess_score_change(timeframe=False, min_score=32)
        
        Note:
            Assessments - Case Management Score Changes
        '''

        where_statement = f"""and latest_date >= {self.q_t1} and earliest_date <= {self.q_t2}""" if timeframe else ''

        min_score_statement = f'''and earliest_score >= {min_score}''' if min_score else ''
        query = f'''
        with tracking_base as(SELECT *, 
        case when earliest_score > latest_score then 'lowered' 
            when earliest_score = latest_score then 'remained'
            else 'increased'
            end as score_change
        FROM assess.cm_tracker
        join(select distinct participant_id from {self.table}) s using(participant_id)
        where total_assess > 1 {where_statement} {min_score_statement})

    select assessment_type,
    count(distinct participant_id) total_reassessed,
    count(distinct case when score_change = 'lowered' then participant_id else null end) lowered,
    count(distinct case when score_change = 'remained' then participant_id else null end) remained,
    count(distinct case when score_change = 'increased' then participant_id else null end) increased
    from tracking_base
    group by assessment_type
        '''
        df = self.query_run(query)
        return(df)
    
    def assess_risk_factor_assessments(self, timeframe = True, distinct_clients = False):
        '''
        Counts # of risk factor, protective, and strength-based inventories for clients.

        Parameters:
            timeframe (Bool): Whether to only count assessments in timeframe. Defaults to True
            distinct_clients (Bool): Whether to count multiple assessments per client. Defaults to False
        
        Note:
            Assessments - Count by Inventory Type (protective, risk-factor, strength-based)
        '''
        distinct = 'distinct ' if distinct_clients else ''
        timeframe = f'and assessment_date between {self.q_t1} and {self.q_t2}' if timeframe else ''

        query = f'''
        with cm as (select participant_id, assessment_date, assessment_type,
        case when assessment_type = 'ASSM' then 'Asset'
			when assessment_type regexp "BP|PCL" then 'Risk'
            else 'Protective' end as assessment_category
        from assess.cm_long
        union all
        select participant_id, assessment_date, 
        'Needs Assessment' assessment_type, 'Protective' assessment_category 
        from assess.needs_assessment_full
                ),
        outreach as (
        select participant_id, assessment_date, 
        'Outreach Eligibility' assessment_type, 'Protective' assessment_category 
        from assess.outreach_eligibility
        union all
        select participant_id, assessment_date, 
        'Safety Assessment' assessment_type, 'Protective' assessment_category 
        from assess.safety_assessment
        union all
        select participant_id, assessment_date, 
        'Safety Plan' assessment_type, 'Protective' assessment_category 
        from assess.safety_intervention)

        select assessment_category, count({distinct}participant_id) count from
        (select * from cm
        union all
        select * from outreach) a
        where participant_id in (select distinct participant_id from {self.table}) {timeframe}
        group by assessment_category
        '''
        df = self.query_run(query)
        return(df)


    @clipboard_decorator
    def custody_status(self, summary_table = False):
        '''
        Returns a table of clients' most recent custody statuses

        Parameters:
            summary_table (Bool): groups clients by latest custody status. Defaults to False

        Examples:
            Get a record of each clients' latest custody status::

                e.custody_status()

            Get the number of clients with each custody status::

                e.custody_status(summary_table=True)
        
        Note:
            Legal - Custody Statuses (Individual or Grouped)
        '''
        
        query = f'''
        with parts as (
        select distinct participant_id, first_name, last_name, program_start
        from {self.table}),
        cust_elig as (select * from parts
        join neon.custody_status using(participant_id)
        join stints.stint_count using(participant_id)
        where (stint_count = 1 or datediff(custody_status_date, program_start) >= -90)),
        cust as (
        select participant_id, custody_status, custody_status_date from cust_elig
        join (select participant_id, max(custody_status_id) custody_status_id from cust_elig 
        group by participant_id)  c
        using(participant_id, custody_status_id)),
        cust_table as(
        select participant_id, first_name, last_name, program_start, case when custody_status is null then 'MISSING' else custody_status end as custody_status, custody_status_date from parts
        left join cust using(participant_id))
        '''
        if summary_table:
            addendum = f'''select custody_status, count(distinct participant_id) as count
                            from cust_table
                            group by custody_status'''
        else:
            addendum = f'''select * from cust_table'''

        query = query + ' ' + addendum

        df = self.query_run(query)
        return(df)

    @clipboard_decorator
    def dem_address(self, new_clients = False, group_by = None):
        '''
        returns client address records. 

        Parameters:
            new_clients (bool): include only new clients. Defaults to False
            group_by: group client records, takes 'zip', 'community', 'region'. Defaults to None.
        
        Examples:
            Get a table of each client's address::

                e.dem_address()
            
            Get a count of the number of client's in each neighborhood::
                
                e.dem_address(group_by='community')
            
            Get a count of the number of new clients in each zipcode::

                e.dem_address(new_clients=True, group_by='zip')
        
        Note:
            Demographics - Client Addresses
        '''
        new_client_statement = f'where program_start between {self.q_t1} and {self.q_t2}' if new_clients else ''
        query = f'''
            with ranked_addresses as (select *,
            ROW_NUMBER() OVER (partition by participant_id ORDER BY primary_address DESC, address_id DESC, civicore_address_id asc) AS rn
            from neon.address
            join (select distinct participant_id, first_name, last_name from {self.table} {new_client_statement}) sn using(participant_id)),
            address_table as(
            select participant_id, first_name, last_name, address1, address2, city, state, zip, primary_address, community
            from ranked_addresses
            where rn = 1)
            '''
        if not group_by:
            modifier = f'''select * from address_table'''
        else:
            if group_by == 'region':
                modifier = f'''
                            ,
                            region_table as(
                select participant_id, case when community not regexp ".*garfield park|little village|north lawndale|austin" then region else community end as community
                from address_table
                join misc.chicago_regions using(community))
                
                select community, count(distinct participant_id) count
                from region_table
                group by community
                ORDER BY 
                    CASE 
                        WHEN community IN ('North Lawndale', 'Austin', 'Little Village', 'East Garfield Park', 'West Garfield Park') THEN 0
                        WHEN community = 'Other_Chicago' THEN 2
                        ELSE 1
                    END,
                    community ASC;
                '''
            else:
                modifier = f'''select {group_by}, count(distinct participant_id) as count from address_table
                group by {group_by}'''
        query = self.query_modify(str(query), modifier)
        df = self.query_run(query)
        return(df)
        
    @clipboard_decorator
    def dem_age(self, new_clients = False, age = 18):
        '''
        Returns a count of clients below/above a certain age threshold, or identifies clients as juveniles/adults 

        Parameters:
            new_clients (Bool): if true, only counts clients who began between t1 and t2. defaults to False
            tally (Bool): if true, returns a count of juv/adults, if false, returns a list. defaults to True
            age: threshold at which a client is counted as a juvenile. defaults to 18

        Examples:
            Get the number of clients currently under 19::

                e.dem_age(age=19)

            Get the number of new clients under 18:

                e.dem_age(new_clients=True)
        
        Note:
            Demographics - Client Ages
        '''
        new_client_condition = f'''WHERE program_start between {self.q_t1} AND {self.q_t2}''' if new_clients else ''

        query = f'''
        with ages as(
            select *, case when age < {age} then 'juvenile' when age between {age} and 25 then 'emerging adult' when age > 25 then 'adult' else 'missing' end as current_age_group,
            case when active_age < {age} then 'juvenile' when active_age  between {age} and 25 then 'emerging adult' when age > 25 then 'adult' else 'missing' end as active_age_group,
            case when stint_age < {age} then 'juvenile' when stint_age  between {age} and 25 then 'emerging adult' when age > 25 then 'adult' else 'missing' end as stint_age_group
            from 
            (select *, 
            floor(datediff(program_start, birth_date)/365) as active_age,
            floor(datediff({self.q_t1}, birth_date)/365) as stint_age
            from {self.table} {new_client_condition})ag),

            grouped_ages as(select current_age_group age_group, count(distinct participant_id) as current_age
            from ages group by current_age_group)

            select age_group, current_age, enrollment_age, stint_start_age
            from grouped_ages
            left join (select active_age_group age_group , count(distinct participant_id) as enrollment_age
            from ages group by age_group) a using(age_group)
            left join (select stint_age_group age_group , count(distinct participant_id) as stint_start_age
            from ages group by age_group) s using(age_group)
            group by age_group
            order by case when age_group ='juvenile' then 1 when age_group ='emerging adult' then 2  when age_group ='adult' then 3 else 4 end'''

        df = self.query_run(query)
        return(df)
    
    @clipboard_decorator
    def dem_race_gender(self, race_gender = 'race',new_clients = False):
        '''
        Returns a count of client races or genders

        Parameters:
            new_clients (Bool): if true, only counts clients who began between t1 and t2. defaults to False
            race_gender: the category to tally, enter either "race" or "gender". defaults to 'race'

        Examples:
            Get the genders of new clients::

                e.dem_race_gender(new_clients=True, race_gender='gender')

            Get client races::
                
                e.dem_race_gender()
        
        Note:
            Demographics - Client Races/Genders
        '''
        query = f'''select {race_gender}, count(distinct participant_id) as count
        from {self.table}
        group by {race_gender}'''
        modifier = f'''WHERE program_start between {self.q_t1} AND {self.q_t2}'''
        if new_clients is True:
            query = self.query_modify(str(query), modifier)  # Use self.query_modify here
        df = self.query_run(query)
        return(df)
    
    @clipboard_decorator
    def enrollment(self, program_type = False, service_type = False, grant_type = False):
        '''
        Returns a count of clients, with options to break down by program, service, and/or grant.

        Parameters:
            program_type(Bool): distinguish by program, defaults to False
            service_type(Bool): distinguish by service, defaults to False
            grant_type(Bool): distinguish by grant, defaults to False

        Examples:
            Get the total number of clients enrolled::

                e.enrollment()

            Get the number of clients enrolled in each program::

                e.enrollment(program_type=True)
            
            Get the number of clients receiving each service for every program::

                e.enrollment(program_type=True, service_type=True)

            Get the number of clients receiving each service on a grant::

                e.enrollment(service_type=True, grant_type=True)
        
        Note:
            Enrollment - Total Clients (overall or by program/service/grant)
        '''

        types_list = []
        if program_type:
            types_list.append('program_type')
        if service_type:
            types_list.append('service_type')
        if grant_type:
            types_list.append('grant_type')
        if types_list:
            types_str = ', '.join(str(st) for st in types_list)
            query = f'''select {types_str}, count(distinct participant_id) as count
            from {self.table}
            group by {types_str}'''
        else:
            query = f'''select count(distinct participant_id) count
            from {self.table}'''
        df = self.query_run(query)
        return(df)

    def enrollment_bundles(self):
        '''
        Counts clients by their bundle of programs

        Example:
            Get the number of clients enrolled in each combination of programs::
                
                e.enrollment_bundles()
        
        Note:
            Enrollment - Client Program Combinations
        '''
        query = f'''select prog, count(distinct participant_id) from
        (select participant_id, group_concat(distinct program_type) prog from stints.neon
        group by participant_id) s
        group by prog'''
        df = self.query_run(query)
        return(df)

    def enrollment_flow(self):
        '''
        Counts the flow of clients in and out of LCLC in the timeframe.

        Example:
            Get the number of clients enrolled/unenrolled in the timeframe::

                e.enrollment_flow()
        
        Note:
            Enrollment - Status Changes
        '''
        
        query = f'''
        with progs as(
            select distinct(program_type), participant_id, first_name, last_name, prog_counts ,program_start, program_end,
            case when program_start between {self.q_t1} and {self.q_t2} then 1 else 0 end as if_started,
            case when program_end between {self.q_t1} and {self.q_t2} then 1 else 0 end as if_ended
            from {self.table}
            join (select participant_id, count(distinct program_type) prog_counts
            from {self.table}
            group by participant_id) p using(participant_id)),
            
        status_counter as (
            select participant_id, prog_counts, sum(if_started) started_progs, sum(if_ended) ended_progs
            from progs
            group by participant_id, prog_counts),
            
        client_statuses as (
            select participant_id,
            case when prog_counts = started_progs and prog_counts = ended_progs then 'started_ended'
            when prog_counts = started_progs then 'started'
            when prog_counts = ended_progs then 'ended'
            else 'continuing'
            end as 'prog_status'
            from status_counter)
            
            select 'TOTAL' as 'status', count(distinct participant_id) count from {self.table}
            union all
            (select prog_status, count(distinct participant_id) count from client_statuses
            group by prog_status)
            union all
            select concat('Close Reason: ', close_reason), count(distinct participant_id) reason_count from client_statuses
            join neon.programs using(participant_id)
            where prog_status like '%ended' and close_reason is not null
            group by close_reason
            order by case when 
            status = 'TOTAL' then 1
            when status = 'continuing' then 2
            when status = 'started' then 3
            when status = 'started_ended' then 4
            when status = 'ended' then 5 else 6 end asc
            ;

        '''

        df = self.query_run(query)
        return df
    
    def incident_tally(self):
        '''
        counts incidents in timeframe, distinguishing between CPIC and non-CPIC events.  

        Example:
            Get the number of incidents in the timeframe::

                e.incident_tally()
        
        Note:
            Mediations/Critical Incidents - Count by Notification Type
        '''

        query = f'''SELECT count(case when how_hear regexp '.*CPIC.*' then incident_id else null end) as CPIC,
	count(case when how_hear not regexp '.*CPIC.*' then incident_id else null end) as non_CPIC
    FROM neon.critical_incidents
    where (incident_date between {self.q_t1} and {self.q_t2})'''
        df = self.query_run(query)
        return df
    
    def incident_type_tally(self):
        '''
        counts incidents in timeframe, distinguishing between fatal and non-fatal events
        
        Note:
            Mediations/Critical Incidents - Count by Incident Type
        '''
        
        query = f'''SELECT type_incident,
            count(case when num_deceased > 0 then incident_id else null end) as fatal,
            count(case when num_deceased = 0 then incident_id else null end) as non_fatal
            FROM neon.critical_incidents
            where incident_date between {self.q_t1} and {self.q_t2}
            group by type_incident'''
        df = self.query_run(query)
        return df


    def incident_response(self):
        '''
        counts incidents responded to in timeframe

        Example:
            Get the number of incident responses in the timeframe::

                e.incident_response()
        
        Note:
            Mediations/Critical Incidents - Count by Response Type
        '''

        query = f'''select count(incident_id) as total_incidents, 
        count(case when did_staff_respond = 'yes' then incident_id else null end) as responded_incidents 
        from neon.critical_incidents
        where (incident_date between {self.q_t1} and {self.q_t2})'''
        df = self.query_run(query)
        return df
    
    @clipboard_decorator
    def isp_goal_tracker(self):
        '''
        Breaks out client ISP goals by domain and completion

        Example:
            Get the status of client ISP goals by domain::

                e.isp_goal_tracker()
        
        Note:
            ISPs - Completion Rate by Goal Area
        '''

        query = f'''SELECT goal_domain, count(case when latest_status = 'in progress' then participant_id else null end) as in_progress,
        count(case when latest_status = 'completed' then participant_id else null end) as completed,
        count(case when latest_status = 'discontinued' then participant_id else null end) as discontinued,
        count(case when latest_status = 'Not Yet Started' then participant_id else null end) as not_yet_started FROM neon.isp_status
        join (select distinct participant_id, service_start from {self.table}
        where service_type like 'case management') i using(participant_id)
        group by goal_domain'''
        df = self.query_run(query)
        return df

    @clipboard_decorator
    def isp_tracker(self, just_cm = True, summary_table = False, service_days_cutoff = 45):
        '''
        Returns a table of client service plan statuses or a table summarizing overall plan completion.

        Parameters:
            just_cm (Bool): if true, only looks at clients enrolled in case management. Defaults to True
            summary_table (Bool): whether to return a summary table. Defaults to False
            service_days_cutoff: the day threshold at which a service plan ought to be complete. Defaults to 45 
    
        Examples:
            Get a full table of ISP statuses for all clients::

                e.isp_tracker(just_cm=False)

            Get the count of case management clients missing a service plan after 60 days::

                e.isp_tracker(summary_table=True, service_days_cutoff=60)
        
        Note:
            ISPs - Status Tables (Individual or Grouped)
        '''

        if just_cm == True:
            prog_serv = 'service'
            base_table = f'''select participant_id, first_name, last_name, datediff({prog_serv}_stop, {prog_serv}_start) as {prog_serv}_days from
            (select participant_id, first_name, last_name, {prog_serv}_start, {prog_serv}_end, case when {prog_serv}_end is null then {self.q_t2} else {prog_serv}_end end as {prog_serv}_stop from {self.table}
            join 
                (select participant_id, {prog_serv}_type, max({prog_serv}_start) as {prog_serv}_start from {self.table}
                where {prog_serv}_type = 'case management' group by participant_id, {prog_serv}_type) st 
            using(participant_id, {prog_serv}_type, {prog_serv}_start)) serv'''
        if just_cm == False:
            prog_serv = 'program'
            base_table = f'''select participant_id, first_name, last_name, datediff({prog_serv}_stop, {prog_serv}_start) as {prog_serv}_days from
            (select distinct participant_id, first_name, last_name, {prog_serv}_start, {prog_serv}_end, case when {prog_serv}_end is null then {self.q_t2} else {prog_serv}_end end as {prog_serv}_stop from {self.table}
            join 
                (select participant_id, max({prog_serv}_start) as {prog_serv}_start from  {self.table}
                group by participant_id, {prog_serv}_type) st 
            using(participant_id, {prog_serv}_start)) serv'''
        
        query_to_modify = f'''
        with base as ({base_table}),
        assms as (select participant_id, assessment_date latest_assm_date, max(sum) as latest_assm_score from assess.assm
            join (select participant_id, max(assessment_date) assessment_date from assess.assm
            group by participant_id) na using(participant_id, assessment_date)
            group by participant_id, assessment_date),
        custody_status as(
            select participant_id, custody_status, custody_status_date from neon.custody_status
            join 
            (select participant_id, max(custody_status_id) as custody_status_id from neon.custody_status
            group by participant_id) ncs using(participant_id, custody_status_id)),
        big_table as(
            select * from base
            left join custody_status using(participant_id)
            left join assms using(participant_id)
            left join neon.isp_tracker using(participant_id))

        '''

        if summary_table == False:
            modifier = '''select * from big_table'''
        
        if summary_table == True:
            modifier = f'''
            ,
            elig as (select *, case when (custody_status = 'In Custody' or {prog_serv}_days <={service_days_cutoff}) or (custody_status is null and {prog_serv}_days <={service_days_cutoff}) then 'optional needs' else 'needs needs' end as eligibility
            from big_table)

            select eligibility, count(distinct participant_id) as total_participants,
            count(case when latest_assm_date is not null and isp_start is not null then participant_id end) as completed_needs,
            count(case when latest_assm_date is not null and isp_start is null then participant_id end) as just_assm,
            count(case when latest_assm_date is null and isp_start is not null then participant_id end) as just_isp,
            count(case when latest_assm_date is null and isp_start is null then participant_id end) as missing_both
            from elig
            group by eligibility

            '''
        query = query_to_modify + ' ' + modifier
        df = self.query_run(query)
        return(df)
    
    @clipboard_decorator
    def isp_discharged(self, missing_names = False):
        '''
        Returns a table of discharged clients' service plan completion and groups percent of goals completed.

        Parameters:
            missing_names (Bool): whether to return a list of the clients missing an ISP. defaults to False
        
        Examples::
            Get the number of discharged clients broken out by % of their service plan completed::

                e.isp_discharged()

            Get a list of discharged clients with no ISPs recorded::

                e.isp_discharged(missing_names=True)
        
        Note:
            ISPs - Completion Rates for Discharged Clients

        '''

        query = f'''
        with discharged_isps as(
        select first_name, last_name, participant_id, isp_start, latest_update, total_goals, goals_completed, goals_in_progress, goals_discontinued, 
        goals_unstarted from neon.isp_tracker
        right join (select * from {self.table} where service_type = 'case management' and (service_end is not null or program_end is not null)) o using(participant_id))
        '''

        if missing_names:
            modifier = f'''
            select first_name, last_name, participant_id from discharged_isps where isp_start is null
            '''
        else:
            modifier = f'''
            ,
        percent_groups as (
        select participant_id, percent_goals_completed, 
        case when percent_goals_completed is null then 'Missing'
            when percent_goals_completed = 0 then '0%'
            when percent_goals_completed > 0 and percent_goals_completed < .5 then '0%-50%'
            when percent_goals_completed > .5 and percent_goals_completed < 1 then '50%-100%'
            else '100%'
            end as percent_group
            from (select participant_id, goals_completed/total_goals as percent_goals_completed
            from discharged_isps) i)
            
            select percent_group percent_complete, count(distinct participant_id) as count from percent_groups
            group by percent_group
            order by case 
            when percent_group = '100%' then 1
            when percent_group = '50%-100%' then 2
            when percent_group = '0%-50%' then 3
            when percent_group = "0%" then 4
            else 5
            end asc
            '''
        query = query + ' ' + modifier
        df = self.query_run(query)
        return(df)

    @clipboard_decorator
    def legal_bonanza(self, timeframe = False, case_stage = None, ranking_method = None, grouping_cols = [], wide_col = None):
        '''
        Flexible function designed to return a table of legal data.


        Parameters:
            timeframe (Bool): Whether to look true only looks at cases active in time period
            case_stage (optional): 'started' only looks at cases started in time period, 'ended' looks at cases ended
            ranking_method (optional): 'highest_felony' looks at a client's highest pretrial charge, 'highest_outcome' looks at a clients highest outcome. Defaults to "None"
            grouping_cols (str, list): column(s) to use group_by on. The string 'case_outcomes' automatically includes case_outcome, sentence, and probation_type. "Total" gives total number of cases.
            wide_col(optional): column to break results out by: 'fel_reduction' - if felony reduced, 'violent' - if case was violent, 'class_type' - if case was felony/misdemeanor
        
        Hints:
            group_by column options: case_type, violent, juvenile_adult, class_prior_to_trial_plea, class_after_trial_plea, case_outcome, sentence, probation_type
            wide_col column options: violent, fel_reduction, class_type

        Examples:
            Get client outcomes in the time period::

                e.legal_bonanza(timeframe=True, ranking_method='highest_outcome', grouping_cols='case_outcomes')

            Get types of cases begun in time period::

                e.legal_bonanza(timeframe=True, case_stage = 'started', grouping_cols='case_type')

            Get case outcomes grouped by violent status::

                e.legal_bonanza(timeframe=True, case_stage='ended', grouping_cols='case_outcome', wide_col='violent')
        
        Note:
            Legal - Flexible Master Function
        '''
        if 'a2j_' in self.table:
            base_table = f'''with base as (
            select * from {self.table} '''
        else:
            base_table = f'''with base as (
            select * from neon.legal_mycase
            join (select distinct participant_id from {self.table}) n using(participant_id)'''
        
        if timeframe is True:
            # where statement exists
            if case_stage is None:
                addendum = f'''where (case_start is null or case_start <= {self.q_t2}) and ((case_outcome_date is null and (case_end is null or case_end > {self.q_t1})) or case_outcome_date between {self.q_t1} and {self.q_t2}))'''
            elif case_stage.lower() == 'started':
                addendum = f'''where case_start between {self.q_t1} and {self.q_t2})'''
            elif case_stage.lower() == 'ended':
                addendum = f'''where case_outcome_date between {self.q_t1} and {self.q_t2})'''
        if timeframe is False:
            addendum = ')'
        
        base_table = base_table + ' ' + addendum

        table_name = 'base'

        if ranking_method:
            ranked_tables = f'''
            , fel_rank as (select *, case when ranking is null then 10 else ranking end better_rank from base b
            left join misc.felony_classes f on b.class_prior_to_trial_plea = f.felony),
            highest_felony as (
            select participant_id, legal_id, mycase_id, mycase_name, case_start, 
            case_end, case_status, case_type, violent, juvenile_adult, class_prior_to_trial_plea, class_after_trial_plea, fel_reduction,
            case_outcome, case_outcome_date, sentence, probation_type, probation_requirements, sentence_length,
            expungable_sealable, expunge_seal_elig_date, expunge_seal_date, was_sentence_reduced, reduction_explain, case_entered_date, outcome_entered_date,
            go_to_trial from fel_rank
            join 
            (select participant_id, min(better_rank) as better_rank from 
            fel_rank group by participant_id) fr using(participant_id, better_rank)),

            highest_outcome as(
            select b.* from 
            (select legal_id,
            ROW_NUMBER() OVER (partition by participant_id ORDER BY case_rank desc, sentence_rank desc) as rn
            from(
            select l.*, c.ranking as case_rank, s.ranking as sentence_rank from base l
            left join misc.highest_sentence s using(sentence)
            left join misc.highest_case c using(case_outcome)) lcs
            where case_outcome is not null) ranked_outcomes
            join base b using(legal_id)
            where rn = 1)
            
                '''
            base_table = base_table + " " + ranked_tables 
            table_name = ranking_method
        
        #handle grouping cols
        existing_groups = {'case_outcomes': ['case_outcome', 'sentence', 'probation_type']}
        if isinstance(grouping_cols, str):
            if grouping_cols in existing_groups:
                grouping_cols = existing_groups[grouping_cols]
            else:
                grouping_cols = list(grouping_cols.split(", "))

        # make wide as desired
        if wide_col:
            count_str = f'''count(distinct case when {wide_col} regexp 'Yes|Felony|reduced' then mycase_id else null end) as '{wide_col}',
            count(distinct case when {wide_col} regexp 'No|Misdemeanor|remained' then mycase_id else null end) as 'not {wide_col}',
            count(distinct case when {wide_col} is null or {wide_col} like "%missing%" then mycase_id else null end) as "missing"'''
        else:
            count_str = 'count(distinct mycase_id) as count'

        if grouping_cols: 
            if grouping_cols[0].lower() == 'total':
                actual_query = f'''
                
                select count(distinct mycase_id) as total_cases
                from {table_name}'''
            
            else:
                cols = ', '.join(str(col) for col in grouping_cols)
                actual_query = f'''

                select {cols}, {count_str}
                from {table_name}
                group by {cols}
                order by {grouping_cols[0]} asc'''
        else:
            actual_query = f'''
            
            select * from {table_name}'''

        query = base_table + ' ' + actual_query

        df = self.query_run(query)
        return(df)
    
    @clipboard_decorator
    def legal_rearrested(self, client_level = True):
        '''
        Returns a count of clients who picked up new cases cumulatively and in the timeframe

        Parameters:
            client_level(Bool): True counts the number of clients, False counts the number of total cases. Defaults to True
        
        Examples:
            Get the number of new cases picked up by clients::

                e.legal_rearrested(client_level=False)
            
            Get the number of clients rearrested::

                e.legal_rearrested()
        
        Note:
            Legal - Recidivism Rates
        '''

        distinct = 'distinct ' if client_level else ''
        query = f'''
        with base as (
        select * from neon.legal_mycase
        join (select distinct participant_id from {self.table}) n using(participant_id)
        where (mycase_name not like "%traffic%" and mycase_id != 31580789)),
        rearrests as (
        select b.*, case when case_start between {self.q_t1} and {self.q_t2} then 'Yes' else 'No' end as timeframe from base b
        join 
        (select participant_id, min(case_start) as earliest_start
        from base
        group by participant_id) e using(participant_id)
        where case_start > earliest_start)

        select * from (select count({distinct} participant_id) as total_clients from base) b
        join (select count({distinct} participant_id) as total_rearrested, count({distinct} case when timeframe = 'Yes' then participant_id else null end) as timeframe_rearrested from rearrests) r
        '''
        df = self.query_run(query)
        return(df)

    @clipboard_decorator
    def legal_rjcc(self, client_level = True,timeframe = True):
        '''
        Returns a count of clients enrolled in RJCC (according to MyCase), and a tally of case outcomes

        Parameters:
            client_level(Bool): True counts the number of clients, False counts the number of total cases. Defaults to True
            timeframe(Bool): True only looks at cases in the timeframe. Defaults to True

        Examples:
            Get RJCC cases ended in timeframe::

                e.legal_rjcc(client_level=False)
            
            Get the number of clients who completed RJCC::

                e.legal_rjcc()
        
        Note:
            Legal - RJCC Enrollment in MyCase
        '''
        base_where = f"and (stage_end is null or stage_end >= {self.q_t1}) and stage_start <= {self.q_t2}" if not timeframe else ''
        final_where = f'where case_outcome_date is null or case_outcome_date >= {self.q_t1}' if not timeframe else ''
        distinct = 'distinct' if client_level else ''

        query = f'''
        with rjcc_base as(select * from mycase.case_stages
        join {self.table} using(participant_id)
        left join neon.legal_mycase using(participant_id, mycase_id, legal_id, mycase_name, case_id)
        where stage = 'rjcc' {base_where})


        select count({distinct} participant_id) total_enrolled,
        count({distinct} case when case_outcome is not null and case_outcome_date <= {self.q_t2}  then participant_id else null end) as total_completed,
        count({distinct} case when case_outcome like "%dismissed%" and case_outcome_date <= {self.q_t2} then participant_id else null end) as successfully_completed,
        count({distinct} case when case_outcome not like "%dismissed%" and case_outcome_date <= {self.q_t2} then participant_id else null end) as unsuccessfully_completed
        from rjcc_base
        {final_where}
        '''
        df = self.query_run(query)
        return(df)

    def linkages_edu_completed(self):
        '''
        Returns completed education linkages in timeframe

        Note:
            Linkages - Completed Education Linkages in Timeframe
        '''
        query = f'''select participant_id, linkage_org, comments from neon.linkages
        join (select distinct participant_id from  {self.table}) n using(participant_id)
        where linkage_type regexp 'education.*' and end_date between {self.q_t1} and {self.q_t2} and end_status = 'Successfully Completed';
        '''
        df = self.query_run(query)
        return(df)

    @clipboard_decorator
    def linkages_edu_employ(self, just_cm = True,first_n_months = None, ongoing = False, age_threshold = 18, new_client_threshold = 45, include_wfd = True):
        '''
        Counts the number of clients enrolled/employed by age group. 

        Parameters:
            just_cm (Bool): Whether to only include clients enrolled in case management. Defaults to True
            first_n_months (optional, int): Only counts linkages in the first N months of program enrollment, usually 6 or 9. Defaults to None
            ongoing (Bool): Only include linkages with no end date. Defaults to False
            age_threshold (int): inclusive upper bound for 'school-aged' clients. Defaults to 18
            new_client_threshold (int): number of days required to be considered "continuing" 
            include_wfd (Bool): whether to count workforce development linkages as employment. Defaults to True.
        
        Examples:
            Get the number of case management clients enrolled/employed in their first 9 months::

                e.linkages_edu_employ(first_n_months=9)

            Get the number of clients currently enrolled/employed with an age cutoff of 19::

                e.linkages_edu_employ(just_cm=False, ongoing=True, age_threshold=19)
            
            Get the number of case management clients enrolled/employed excluding workforce development linkages::

                e.linkages_edu_employ(include_wfd=False)
            
            Get the number of case management clients enrolled/employed after nine months::

                e.linkages_edu_employ(new_client_threshold = 275)
        
        Note:
            Linkages - Big Education and Employment Table
        '''

        workforce = '|Workforce Development' if include_wfd else ''
        start_date = 'service_start' if just_cm else 'program_start'
        if first_n_months:
            new_client_threshold = int(first_n_months * 30.5) if new_client_threshold == 45 else new_client_threshold
        query = f'''
        with part as (
        select participant_id, 
        case when timestampdiff(year, birth_date, {start_date}) <= {age_threshold} then 'juvenile' when timestampdiff(year, birth_date, {start_date}) > {age_threshold} then 'adult' else 'missing' end as age_group, 
        program_start, service_start, case when datediff({self.q_t2}, {start_date}) > {new_client_threshold} then 'cont' else 'new' end as newness 
        from {self.table}
        {f"where service_type = 'Case Management'" if just_cm else ''}),
        cust as(
        select participant_id, custody_status from neon.custody_status
        join (select participant_id, program_start, service_start,max(custody_status_date) as custody_status_date from part
        join neon.custody_status using(participant_id)
        group by participant_id, program_start, service_start) cs using (participant_id, custody_status_date)
        where datediff(custody_status_date, program_start) >-60 and custody_status = 'in custody'),
        intake_info as (
        select participant_id, currently_enrolled, currently_employed from neon.intake
        join (select participant_id, max(intake_date) as intake_date from neon.intake
        group by participant_id) i using(participant_id, intake_date)
        join part using(participant_id)
        where intake_date >= program_start),
        base as (
        select * from part
        left join intake_info using(participant_id)
        left join cust using(participant_id)),
        link_tally as (
        select participant_id, count(case when linkage_type like 'education%' then linkage_id else null end) as edu_links,
        count(case when linkage_type regexp 'employment{workforce}' then linkage_id else null end) as job_links
        from part
        join neon.linkages using(participant_id)
        where program_start <= linked_date and linkage_type regexp "education.*|employment{workforce}" and start_date is not null
        {f"and end_date is null" if ongoing else ''} 
        {f"and DATEDIFF(linked_date, program_start) <= {first_n_months} * 30.5" if first_n_months else ''}
        group by participant_id),
        link_base as (
        select * from base
        left join link_tally using(participant_id))

        select age_group, newness, custody_status, count(distinct participant_id) as total_clients,
        {f"count(distinct case when DATEDIFF({self.q_t2}, {start_date}) >= {first_n_months} * 30.5 then participant_id else null end) as enrolled_for_{first_n_months}," if first_n_months else ''}
        count(distinct case when currently_enrolled = 'yes' then participant_id else null end) as began_enrolled,
        count(distinct case when edu_links > 0 then participant_id else null end) as school_links,
        count(distinct case when (currently_enrolled is null or currently_enrolled = 'no') and edu_links > 0 then participant_id else null end) as newly_enrolled,
        count(distinct case when currently_employed = 'yes' then participant_id else null end) as began_employed,
        count(distinct case when job_links > 0 then participant_id else null end) as job_links,
        count(distinct case when (currently_employed is null or currently_employed = 'no') and job_links > 0 then participant_id else null end) as newly_employed
        from link_base
        group by age_group, newness, custody_status
        order by custody_status asc, newness asc
        '''
        df = self.query_run(query)
        return(df)

    def linkages_edu_employ_new(self, cm_only = False):
        '''
        Returns new education/employment linkages in timeframe

        Note:
            Linkages - New Education and Employment Connections
        '''
        query = f'''select linkage_type, count(distinct participant_id) participants from neon.linkages
        join (select distinct participant_id from {self.table}
            {f"where service_type = 'Case Management'" if cm_only else ''}) n using(participant_id)
        where linkage_type regexp 'education.*|employ.*|work.*' and start_date between {self.q_t1} and {self.q_t2}
        group by linkage_type'''

        df = self.query_run(query)
        return(df)

    def linkages_isp_goals(self, lclc_initiated = True, timeframe_only = False, discharged_only = False, idhs_edu_employ = False):
        '''
        Returns a table of the number of clients with an active, concluded, or unstarted linkage for an ISP goal domain. Also has a row for # of clients who have a linkage for at least one goal  
        
        Parameters:
            lclc_initiated: only include non-client-initiated linkages. Defaults to True
            timeframe_ony: only include linkages made in the timeframe. Defaults to False
            discharged_only: only include discharged clients. Defaults to False
            idhs_edu_employ: count education linkages for employment goals. Defaults to False

        Note:
            Linkages - Connections by ISP Goal Area
        '''

        query = f'''
        with goal_key as(select distinct goal_domain,
  case when goal_domain regexp 'education.*|.*school' then concat('education.*|',goal_domain)
  when goal_domain like 'employment' then concat('employment.*|workforce.*{'education.*|' if idhs_edu_employ else ''}|',goal_domain)
  when goal_domain like 'mobility' then concat('transportation|',goal_domain)
  else concat(goal_domain, '.*') end goal_regexp
  from neon.isp_goals),

  parts as (select participant_id, min(program_start) program_start from {self.table}
  {f'where service_type = "case management" and program_end between {self.q_t1} and {self.q_t2}' if discharged_only else ''}
   group by participant_id),

goal_areas as (select distinct goal_domain, participant_id, goal_regexp from parts
 join neon.isp_goals using(participant_id)
 join (select participant_id, max(isp_id) isp_id from neon.isp_goals group by participant_id) i using(participant_id, isp_id)
 join goal_key using(goal_domain)),

long_link as (select participant_id, linkage_id, 
  case when internal_program = 'therapy' then 'Mental Health'
  when internal_program = 'housing' then 'Housing' else linkage_type end linkage_type,
  linkage_org, 
  case when start_date is null and (end_date is null or end_date > {self.q_t2}) then 'unstarted'
  when start_date is not null and (end_date is null or end_date > {self.q_t2}) then 'active'
  else 'concluded' end linkage_status
  from neon.linkages
join parts using(participant_id)
where (program_start <= start_date or program_start <= linked_date) and linkage_type is not null
{f'and linked_date between {self.q_t1} and {self.q_t2}' if lclc_initiated else ''} {f'and client_initiated = "no"' if lclc_initiated else ''}),

extra_long_link as (
  select * from long_link
  union all
  select participant_id, linkage_id, case when goal_domain like "education%" then 'Education' else goal_domain end as linkage_type, linkage_org,
  case when start_date is null and (end_date is null or end_date > {self.q_t2}) then 'unstarted'
  when start_date is not null and (end_date is null or end_date > {self.q_t2}) then 'active'
  else 'concluded' end linkage_status from neon.isp_goals_linkages
  join neon.isp_goals using(goal_id)
  join neon.linkages using(linkage_id, participant_id)
where linkage_id not in (select distinct linkage_id from long_link)),

grouped_link as (
select participant_id, linkage_type,
  count(distinct case when linkage_status = 'active' then linkage_id else null end) active,
  count(distinct case when linkage_status = 'concluded' then linkage_id else null end) concluded,
  count(distinct case when linkage_status = 'unstarted' then linkage_id else null end) unstarted
  from extra_long_link
group by participant_id, linkage_type),

  
goals_w_link as (select participant_id, case when goal_domain like "education%" then "Education" else goal_domain end goal_domain, 
  sum(active) active, sum(concluded) concluded, sum(unstarted) unstarted from goal_areas 
left join (
    select * from goal_areas
    left join grouped_link using(participant_id)
    where linkage_type regexp goal_regexp) g 
  using(participant_id, goal_domain, goal_regexp)
group by participant_id, goal_domain),

linkage_counts as(select goal_domain, 
  count(distinct participant_id) total_participants,
  count(distinct case when unstarted is not null then participant_id else null end) linkage_made,
  count(distinct case when active > 0 or concluded > 0 then participant_id else null end) linkage_started,
  count(distinct case when active > 0 then participant_id else null end) linkage_active
from goals_w_link
group by goal_domain),

grouped_linkage_counts as(select goal_domain, 
  count(distinct participant_id) total_participants,
  count(distinct case when unstarted is not null then participant_id else null end) linkage_made,
  count(distinct case when active > 0 or concluded > 0 then participant_id else null end) linkage_started,
  count(distinct case when active > 0 then participant_id else null end) linkage_active
from goals_w_link
group by goal_domain
  order by linkage_made desc),

total_row as(select 'ANY DOMAIN' goal_domain, count(distinct participant_id) total_participants, 
  count(distinct case when unstarted is not null then participant_id else null end) linkage_made,
  count(distinct case when active > 0 or concluded > 0 then participant_id else null end) linkage_started,
  count(distinct case when active > 0 then participant_id else null end) linkage_active
  
  from goals_w_link)

select goal_domain, total_participants, 
  round(linkage_active/total_participants, 2) linkage_active,
  round(linkage_started/total_participants, 2) linkage_started,
  round(linkage_made/total_participants, 2) linkage_made
from (select * from total_row 
      union all
      select * from grouped_linkage_counts) t
        '''

        df = self.query_run(query)
        return(df)
    
    @clipboard_decorator
    def linkages_monthly(self, lclc_initiated = True, just_cm = False):
        '''
        Counts the number of clients linked in the current time frame, and in their first 3/6/9 months

        Parameters:
            lclc_initiated (Bool): Only look at linkages that LCLC initiated. Defaults to True
            just_cm (Bool): Only look at clients receiving case management. Defaults to True

        Examples::
            Get the number of case management clients with lclc-initiated linkages in the current time period and their first 3/6/9 months::
            
                e.linkages_monthly()

            Get the number of all clients with linkages in the time period::

                e.linkages_monthly(just_cm=True)
            
            Get the number of case management clients with linkages in the time periods, including client-initiated linkages::

                e.linkages_monthly(lclc_initiated=False)
        
        Note:
            Linkages - Number of Clients Linked in Time Period
        '''


        service_statement = "where service_type = 'case management'" if just_cm else ''
        initiated_statement = "and client_initiated = 'no'" if lclc_initiated else ''
        
        query = f'''with parts as  (select distinct(participant_id), program_start from {self.table}
        {service_statement}),
        base as (select *, timestampdiff(month, program_start, linked_date) month_diff, timestampdiff(month, linked_date, {self.q_t2}) recent_diff from neon.linkages
        join parts using(participant_id)
        where (linked_date > program_start or start_date > program_start) and linked_date <= {self.q_t2} {initiated_statement}),
        recent_links as (
        select participant_id, count(linkage_id) total_links,
        count(case when month_diff < 4 then linkage_id else null end) first_3,
        count(case when month_diff < 7 then linkage_id else null end) first_6,
        count(case when month_diff < 10 then linkage_id else null end) first_9,
        count(case when recent_diff < 1 then linkage_id else null end) this_month
        from base
        group by participant_id)

        select count(distinct participant_id) as total_clients,
        count(case when total_links > 0 then participant_id else null end) as has_links,
        count(case when first_3 > 0 then participant_id else null end) as first_3,
        count(case when first_6 > 0 then participant_id else null end) as first_6,
        count(case when first_9 > 0 then participant_id else null end) as first_9,
        count(case when this_month > 0 then participant_id else null end) as this_month
        from parts
        left join recent_links using(participant_id)'''
        df = self.query_run(query)
        return(df)
    

    @clipboard_decorator
    def linkages_percent(self, timeframe = True, new_client_threshold = 45, cm_only = True, first_n_months = None):
        '''
        Get percent of clients with linkage, broken out by custody/newness

        Parameters:
            timeframe (Bool): Only include records with a linked_date in the timeframe. Defaults to True
            new_client_threshold (int): number of days required to be considered "continuing". Defaults to 45
            cm_only (Bool): Only count case management clients. Defaults to True
            first_n_months (int): only count linkages received in a clients first N months. Defaults to None
        
        Note:
            Linkages - Percent of Clients with a Connection
        '''
        timeframe_statement = f'and linked_date between {self.q_t1} and {self.q_t2}' if timeframe else ''
        cm_only_statement = "where service_type = 'Case Management'" if cm_only else ''
        first_n_months_statement = f"and (datediff(linked_date, service_start) <= 30.5*{first_n_months})" if first_n_months else ''

        query = f'''
        with part as (
        select participant_id, age, program_start, service_start,case when datediff({self.q_t2}, service_start) > {new_client_threshold} then 'cont' else 'new' end as newness
        from {self.table}
        {cm_only_statement}),
        cust as(
        select participant_id, custody_status from neon.custody_status
        join (select participant_id, program_start, service_start,max(custody_status_date) as custody_status_date from part
        join neon.custody_status using(participant_id)
        group by participant_id, program_start, service_start) cs using (participant_id, custody_status_date)
        where datediff(custody_status_date, program_start) >-60 and custody_status = 'in custody'),
        base as (
        select * from part
        left join cust using(participant_id)),
        link_tally as (
        select participant_id, count(linkage_id) as link_count
        from part
        join neon.linkages using(participant_id)
        join stints.stint_count using(participant_id)
        where (program_start <= linked_date or stint_count = 1) and client_initiated = 'No'
        {timeframe_statement} {first_n_months_statement}
        group by participant_id),
        link_base as (
        select * from base
        left join link_tally using(participant_id))

        select *, linked_clients/total_clients linked_percentage from
        (select newness, custody_status, count(distinct participant_id) total_clients, 
        count(distinct case when link_count is not null then participant_id end) linked_clients from link_base
        group by newness, custody_status) s'''

        df = self.query_run(query)
        df = df.reset_index(drop=True).sort_values(by=['linked_percentage'], ascending=False)
        df['linked_percentage'] = df['linked_percentage'].apply(lambda x: "{:.2f}%".format(x * 100))
        return(df)

    @clipboard_decorator
    def linkages_tally(self, lclc_initiated = True, just_cm = False, timeframe = True, distinct_clients = False, group_by = 'linkage_type',link_started = False, link_ongoing = False):
        '''
        Flexible function designed to return linkage information grouped in some way


        Parameters:
            lclc_initiated (Bool): Only look at linkages that LCLC initiated. Defaults to True
            just_cm (Bool): Only look at clients receiving case management. Defaults to True
            timeframe (Bool): Only include records with a linked_date in the timeframe. Defaults to True
            distinct_clients (Bool): Whether only one record should be counted per client. Defaults to False
            group_by (str): The column to group records by (linkage_type, internal_external, linkage_org). Defaults to 'linkage_type'
            link_started (Bool): Only include linkages with a start date. Defaults to False
            link_ongoing (Bool): Only include linkages with no end date. Defaults to False

        Examples:
            Get the types of linkages recorded for all clients in the timeframe::
                
                e.linkages_tally()
            
            Get the number of case management clients with an internal/external linkage at any time::
            
                e.linkages_tally(just_cm=True, timeframe=False, distinct_clients=True, group_by='internal_external')
            
            Get the number of clients with a started linkage of each type in the timeframe::

                e.linkages_tally(distinct_clients=True, link_started=True)
            
            Get the number of clients linked to organizations at any time, with the linkage currently ongoing::

                e.linkages_tally(distinct_clients=True, group_by='linkage_org', link_started=True, link_ongoing=True)
        
        Note:
            Linkages - Flexible Master Function
        '''

        service_statement = "where service_type = 'case management'" if just_cm else ''
        initiated_statement = "and client_initiated = 'no'" if lclc_initiated else ''
        
        parameters_list = [f'linked_date between {self.q_t1} and {self.q_t2}' if timeframe else None, 
                    'start_date is not null' if link_started else None, 'end_date is null' if link_ongoing else None]
        better_parameters = 'where ' + ' and '.join(filter(None, parameters_list)) if any(parameters_list) else ''

        query = f'''with parts as  (select distinct(participant_id) from {self.table}
        {service_statement}),
   base as (select *, timestampdiff(month, stint_start, linked_date) month_diff, timestampdiff(month, linked_date, {self.q_t2}) recent_diff from neon.linkages
        join parts using(participant_id)
        join (select * from stints.stints_plus_stint_count
        join (SELECT participant_id, max(stint_num) stint_num FROM stints.stints_plus_stint_count group by participant_id) s using(participant_id, stint_num)) st using(participant_id)
        where (stint_num = 1 or (linked_date > stint_start or start_date > stint_start)) and linked_date <= {self.q_t2} {initiated_statement}),
        better_base as(select participant_id, 
        case when linkage_type is null and internal_program is not null then concat('LCLC - ', internal_program) else linkage_type end as linkage_type, 
        internal_external, linkage_org, linked_date, start_date, end_date, comments from base
        {better_parameters})
        select {group_by}, count({'distinct ' if distinct_clients else ''}participant_id) count
        from better_base
        {f'group by {group_by}' if group_by else ''}
        '''
        df = self.query_run(query)
        return(df)


    @clipboard_decorator
    def linkages_total(self, lclc_initiated = True, just_cm = False, timeframe = True, distinct_clients = False, link_started = False, link_ongoing = False):
        '''
        Returns the total count of participants linked/linkages made


        Parameters:
            lclc_initiated (Bool): Only look at linkages that LCLC initiated. Defaults to True
            just_cm (Bool): Only look at clients receiving case management. Defaults to True
            timeframe (Bool): Only include records with a linked_date in the timeframe. Defaults to True
            distinct_clients (Bool): Whether only one record should be counted per client. Defaults to False
            link_started (Bool): Only include linkages with a start date. Defaults to False
            link_ongoing (Bool): Only include linkages with no end date. Defaults to False

        Note:
            Linkages - Total Connections Made
        '''
        service_statement = "where service_type = 'case management'" if just_cm else ''
        
        parameters_list = [f'linked_date between {self.q_t1} and {self.q_t2}' if timeframe else None, 
                    'start_date is not null' if link_started else None, 
                    'end_date is null' if link_ongoing else None,
                    'client_initiated = "no"' if lclc_initiated else None]
        better_parameters = 'where ' + ' and '.join(filter(None, parameters_list)) if any(parameters_list) else ''

        query = f'''with parts as  (select distinct(participant_id) from {self.table}
        {service_statement}),
        links as (select * from neon.linkages
        join parts using(participant_id)
        {better_parameters} )

        select {f"'total' {'participants' if distinct_clients else 'linkages'}"}, count(distinct {'participant_id' if distinct_clients else 'linkage_id'}) count
        from links
        '''
        df = self.query_run(query)
        return(df)


    @clipboard_decorator
    def outreach_elig_tally(self, outreach_only = True, new_clients = False):
        '''
        Counts the number of clients with outreach eligibility forms, and the number who answered yes to each question

        Parameters: 
            outreach_only (Bool): Only include clients currently in outreach. Defaults to True
            new_clients (Bool): Only include new clients. Defaults to False

        Examples:
            Get the number of new outreach clients with eligibility screenings/number of "yes"es for each question::

                e.outreach_elig_tally(new_clients=True)
            
            Get the number of all clients with eligibility screenings//number of "yes"es for each question::

                e.outreach_elig_tally(outreach_only=False)
        
        Note:
            Outreach - Eligibility Form Responses
        '''
        where_statement = f'''where service_type = 'outreach''' if outreach_only else ''
        where_statement = f'''where program_start between {self.q_t1} and {self.q_t2} and program_type regexp "chd.*|community navigation.*|violence.*"''' if new_clients else ''


        if new_clients and outreach_only: 
            where_statement = f'''where service_type = 'outreach' and service_start between {self.q_t1} and {self.q_t2}'''

        new = f'''and service_start between {self.q_t1} and {self.q_t2}''' if new_clients else ''
        query = f'''
        select count(distinct participant_id) Total,
        count(distinct case when ASI = 'yes' then participant_id else null end) ASI,
        count(distinct case when JSI = 'yes' then participant_id else null end) JSI,
        count(distinct case when RVV = 'yes' then participant_id else null end) RVV,
        count(distinct case when ORVV = 'yes' then participant_id else null end) ORVV,
        count(distinct case when VAB = 'yes' then participant_id else null end) VAB,
        count(distinct case when FASI = 'yes' then participant_id else null end) FASI,
        count(distinct case when violent_felony_conviction = 'yes' then participant_id else null end) violent_felony_conviction,
        count(distinct case when known_potential_safety_concerns = 'yes' then participant_id else null end) known_safety_concerns,
        count(distinct case when experienced_trauma = 'yes' then participant_id else null end) experienced_trauma,
        count(distinct case when benfit_from_outreach = 'yes' then participant_id else null end) benfit_from_outreach
        from assess.outreach_eligibility
        join (select participant_id, max(assessment_date) assessment_date from assess.outreach_eligibility
        group by participant_id) o using(participant_id, assessment_date)
        join(select distinct participant_id from {self.table} {where_statement}) s using(participant_id)
        '''
        df = self.query_run(query)
        return df
    
    @clipboard_decorator
    def mediation_tally(self,timeframe=True):
        '''
        Counts mediations

        Parameters:
            timeframe (Bool): Whether to only include mediations in the timeframe. Defaults to True
        
        Note:
            Mediations/Critical Incidents - Count
        '''

        ''''''
        timeframe_statement = f"""where mediation_start between{self.q_t1} and {self.q_t2}""" 
        query = f'''select mediation_outcome, count(mediation_id) count
        from neon.mediations 
        {timeframe_statement} 
        group by mediation_outcome
        '''
        df = self.query_run(query)

        return df

    @clipboard_decorator
    def session_tally (self, session_type = 'Case Management', distinct_participants=True):
        '''
        Tallies the number of sessions, or number of clients in the timeframe

        Parameters:
            session_type: the type of session to count. Defaults to 'Case Management', but could also be 'Outreach'
            distinct_participants (Bool): True counts the number of clients, False counts the number of sessions. Defaults to True
        
        Examples:
            Get the number of clients with a case management session in the timeframe::

                e.session_tally()
            
            Get the number of outreach sessions in the timeframe::

                e.session_tally(session_type='Outreach', distinct_participants=False)
        
        Note:
            Case Sessions - CM or Outreach Session Tally
        '''

        session_type = f"'{session_type}'"
        distinct = 'distinct' if distinct_participants else ''
        query = f'''
        with parts as (
        select distinct(participant_id) from {self.table}
        where service_type like {session_type})

        select count(distinct participant_id) as total_parts, 
        count({distinct} case when session_date between {self.q_t1} and {self.q_t2} then participant_id end) as session_attempts,
        count({distinct} case when (session_date between {self.q_t1} and {self.q_t2}) and successful_contact = 'yes' then participant_id end) as successful_sessions
        from parts
        left join (select * from neon.case_sessions where contact_type like {session_type}) c using(participant_id);
        '''
        df = self.query_run(query)
        return(df)
    
    @clipboard_decorator
    def session_frequency(self, session_type = 'Case Management'):
        '''
        Calculates the regularity of client sessions. 

        Parameters:
            session_type: the type of session to count. Defaults to 'Case Management', but could also be 'Outreach'
        
        Examples:
            Get the session regularity of case management clients::
                
                e.session_frequency()
            
            Get the session regularity of outreach clients::

                e.session_frequency(session_type="Outreach")
        
        Note:
            Case Sessions - CM or Outreach Session Frequency
        '''

        session_type = f"'{session_type}'"
        query = f'''
        with datez as (select *, datediff(service_end, service_start) as case_length from 
        (select participant_id, service_start, case when (service_end is null or service_end > {self.q_t2}) then {self.q_t2} else service_end end as service_end from {self.table}
        where service_type = {session_type}) cm),
        sess_counts as (
        select participant_id, count(participant_id) as session_count from neon.case_sessions
        join datez using(participant_id)
        where successful_contact = 'yes' and contact_type = {session_type} and session_date between service_start and service_end
        group by participant_id),
        freq as (
        select participant_id, case 
            when session_frequency <= 7 then "weekly"
            when session_frequency between 7 and 14 then "every two weeks"
            when session_frequency between 14 and 30.4 then "monthly"
            when session_frequency between 30.4 and 60.8 then "every two months"
            when session_frequency > 60.8  then 'less than every two months'
            when session_frequency is null then 'no sessions recorded'
            end as session_rate
        from (select participant_id, case_length, session_count, case_length/session_count as session_frequency from datez
        left join sess_counts using(participant_id)) d)

        select session_rate, count(participant_id) as count
        from freq
        group by session_rate
        order by case when session_rate = 'weekly' then 1
        when session_rate = 'every two weeks' then 2
        when session_rate = 'monthly' then 3
        when session_rate = 'every two months' then 4
        when session_rate = 'less than every two months' then 5
        else 6
        end asc
        '''
        df = self.query_run(query)
        return(df)

class ReferralAsks(Queries):
    def __init__(self, t1, t2, engine, print_SQL = True, clipboard = False, default_table="stints.neon", mycase = True):
        super().__init__(t1, t2, engine, print_SQL, clipboard, default_table, mycase)
    
    def highest_cases(self, attorneys = False):
        '''
        Finds the number of cases for each felony class by team

        Parameters:
            attorneys (Bool): whether to group by attorney or team. Defaults to False

        Examples:
            Get the number cases for each team broken out by highest charge per client::
                
                e.highest_cases()
            
            Get the number cases for each attorney broken out by highest charge per client::

                e.highest_cases()
        '''

        group_by = 'attorneys' if attorneys else 'participant_team'

        query = f'''with base as (
        select * from neon.legal_mycase
        join (select distinct participant_id from {self.table}) n using(participant_id) where (case_start is null or case_start <= {self.q_t2}) and (case_outcome_date is null or case_outcome_date >= {self.q_t1})) 
            , fel_rank as (select *, case when ranking is null then 10 else ranking end better_rank from base b
            left join misc.felony_classes f on b.class_prior_to_trial_plea = f.felony),
            
            highest_felony as (
            select participant_id, legal_id, mycase_id, mycase_name, case_start, 
            case_end, case_status, case_type, violent, juvenile_adult, class_prior_to_trial_plea, class_after_trial_plea, fel_reduction,
            case_outcome, case_outcome_date, sentence, probation_type, probation_requirements, sentence_length,
            expungable_sealable, expunge_seal_elig_date, expunge_seal_date, was_sentence_reduced, reduction_explain, case_entered_date, outcome_entered_date,
            go_to_trial, attorneys, participant_team ,program_start from fel_rank
            join 
            (select participant_id, min(better_rank) as better_rank from 
            fel_rank group by participant_id) fr using(participant_id, better_rank)
            join (select participant_id, attorneys, participant_team from neon.client_teams) ct using(participant_id))

            
        select {group_by}, 
        count(distinct case when class_prior_to_trial_plea = 'Felony X' then mycase_id else null end) as 'Felony X',
        count(distinct case when class_prior_to_trial_plea = 'Felony 1' then mycase_id else null end) as 'Felony 1',
        count(distinct case when class_prior_to_trial_plea regexp 'Felony 2|Felony 3|Felony 4' then mycase_id else null end) as 'Felony 2-4',
        count(distinct case when class_prior_to_trial_plea = 'Misdemeanor A' then mycase_id else null end) as 'Misdemeanor A',
        count(distinct mycase_id) "Total"
        from highest_felony
        group by {group_by}
        
        union all
        
        select "Total",
		count(distinct case when class_prior_to_trial_plea = 'Felony X' then mycase_id else null end) as 'Felony X',
        count(distinct case when class_prior_to_trial_plea = 'Felony 1' then mycase_id else null end) as 'Felony 1',
        count(distinct case when class_prior_to_trial_plea regexp 'Felony 2|Felony 3|Felony 4' then mycase_id else null end) as 'Felony 2-4',
        count(distinct case when class_prior_to_trial_plea = 'Misdemeanor A' then mycase_id else null end) as 'Misdemeanor A',
        count(distinct mycase_id) "Total"
        from highest_felony
        ;'''

        df = self.query_run(query)
        return(df)
    
    def cm_closures(self):
        '''
        Returns the number of closed clients in the timeframe for each case manager

        Example:
            Get the number of case management closures in the timeframe::

                e.cm_closures()
        '''
        
        query = f'''with staff_refs as (select participant_id, full_name, 
        case when a.staff_type like 'outreach%' then 'outreach'
        when a.staff_type like 'case%' then 'case management'
        when a.staff_type like '%attorney%' then 'legal'
        end as 'service_type', staff_start_date, staff_end_date
        from neon.assigned_staff a)

        select full_name as staff_name, count(distinct participant_id) as closed_clients from neon.big_psg
        left join staff_refs using(participant_id, service_type)
        where service_end > {self.q_t1} and service_type = 'Case Management'
        group by full_name'''
        df = self.query_run(query)
        return(df)
    
    def cpic_summary(self):
        '''
        Returns a summary of CPIC notifications in the timeframe

        Example:
            Get a CPIC notification summary::

                e.cm_closures()
        '''

        query = f'''select incident_date, time_notification time_of_notification, did_staff_respond, how_hear, how_notified, num_individuals, num_living, num_deceased, 
        retaliation_future as future_retaliation_potential, comments
        from neon.critical_incidents
        where notification_date between {self.q_t1} and {self.q_t2}'''
        df = self.query_run(query)
        return(df)
    
    def missing_isp(self):
        '''
        Returns a table of clients missing an ISP

        Example:
            Get a table of ISP-less clients::

                e.missing_isp()
        '''
        query = f'''select participant_id, first_name, last_name, case_manager, cm_start, linkage_count, latest_linkage, isp_start from neon.cm_summary
        join (select distinct participant_id from {self.table} where program_end is null and service_end is null) s using(participant_id)
        where isp_update is null
        order by linkage_count desc'''
        df = self.query_run(query)
        return(df)

    def last_30_days(self, successful_sessions = True):
        '''
        Returns a table of clients without a session in the last 30 days

        Parameters:
            successful_sessions (Bool): Only include sessions where contact was successfully made. Defaults to True
        
        Examples:
            Get a record of clients without a CM session in the last 30 days::

                e.last_30_days()
            
            Get a record of clients without a CM session attempt in the last 30 days::

                e.last_30_days(False)
        '''

        sessions_attempts = 'latest_session' if successful_sessions else 'latest_attempt'
        query = f'''select participant_id, first_name, last_name, case_manager, cm_start, latest_linkage, isp_update latest_isp_update, {sessions_attempts} from neon.cm_summary
        where (latest_linkage is null or datediff({self.q_t1}, latest_linkage) >31)
        and (isp_update is null or datediff({self.q_t1}, isp_update) >31)
        and (latest_session is null or datediff({self.q_t1}, {sessions_attempts}) >31)'''
        df = self.query_run(query)
        return(df)


class Grants(Queries):
    def __init__(self, t1, t2, engine, print_SQL = True, clipboard = False, default_table="stints.neon",mycase = True,cloud_run=False,grant_type='idhs'):
        '''
        grant_types: 'idhs', 'idhs_r', 'r3', 'scan', 'ryds'
        '''
        
        super().__init__(t1, t2, engine, print_SQL, clipboard, default_table, mycase,cloud_run)
        if not cloud_run:
            if isinstance(grant_type,str):
                self.table_update(grant_type, update_default_table = True)

    def cvi_demographics(self):
        '''
        Returns Demographics for the ICJIA - CVI report

        Note:
            Grants: ICJIA - CVI Demographics
        '''
        query = f'''with ages as(
        select 'age','0-5' as age_range, count(distinct case when new_client = 'continuing' then participant_id else null end) 'continuing',
count(distinct case when new_client = 'new' then participant_id else null end) 'new',
count(distinct case when discharged_client is not null then participant_id else null end) 'discharged'
        from participants.cvi
        where TIMESTAMPDIFF(YEAR, birth_date, grant_start) BETWEEN 0 AND 5
        UNION ALL
        select 'age','6-11' as age_range, count(distinct case when new_client = 'continuing' then participant_id else null end) 'continuing',
count(distinct case when new_client = 'new' then participant_id else null end) 'new',
count(distinct case when discharged_client is not null then participant_id else null end) 'discharged'
        from participants.cvi
        where TIMESTAMPDIFF(YEAR, birth_date, grant_start) BETWEEN 11 AND 13
        UNION ALL
        select 'age','12-14' as age_range, count(distinct case when new_client = 'continuing' then participant_id else null end) 'continuing',
count(distinct case when new_client = 'new' then participant_id else null end) 'new',
count(distinct case when discharged_client is not null then participant_id else null end) 'discharged'
        from participants.cvi
        where TIMESTAMPDIFF(YEAR, birth_date, grant_start) BETWEEN 12 AND 14
        UNION ALL
        select 'age','15-17' as age_range, count(distinct case when new_client = 'continuing' then participant_id else null end) 'continuing',
count(distinct case when new_client = 'new' then participant_id else null end) 'new',
count(distinct case when discharged_client is not null then participant_id else null end) 'discharged'
        from participants.cvi
        where TIMESTAMPDIFF(YEAR, birth_date, grant_start) BETWEEN 14 AND 17
        UNION ALL
        select 'age','18-25' as age_range, count(distinct case when new_client = 'continuing' then participant_id else null end) 'continuing',
count(distinct case when new_client = 'new' then participant_id else null end) 'new',
count(distinct case when discharged_client is not null then participant_id else null end) 'discharged'
        from participants.cvi
        where TIMESTAMPDIFF(YEAR, birth_date, grant_start) BETWEEN 18 AND 25
        UNION ALL
        select 'age','26+' as age_range, count(distinct case when new_client = 'continuing' then participant_id else null end) 'continuing',
count(distinct case when new_client = 'new' then participant_id else null end) 'new',
count(distinct case when discharged_client is not null then participant_id else null end) 'discharged'
        from participants.cvi
        where TIMESTAMPDIFF(YEAR, birth_date, grant_start) > 25
        UNION ALL 
        select 'age','MISSING' as age_range, count(distinct case when new_client = 'continuing' then participant_id else null end) 'continuing',
count(distinct case when new_client = 'new' then participant_id else null end) 'new',
count(distinct case when discharged_client is not null then participant_id else null end) 'discharged'
        from participants.cvi
        where birth_date is null),
        genders as(
        select 'gender' as 'category',gender, count(distinct case when new_client = 'continuing' then participant_id else null end) 'continuing',
count(distinct case when new_client = 'new' then participant_id else null end) 'new',
count(distinct case when discharged_client is not null then participant_id else null end) 'discharged'
from participants.cvi
group by gender),
races as (select 'race' as 'category',race, count(distinct case when new_client = 'continuing' then participant_id else null end) 'continuing',
count(distinct case when new_client = 'new' then participant_id else null end) 'new',
count(distinct case when discharged_client is not null then participant_id else null end) 'discharged'
from participants.cvi
group by race)

select * from races
union all
select * from genders
union all
select * from ages'''
        df = self.query_run(query)
        return df
    
    def cvi_mental_health_linkages(self):
        '''
        ICJIA - CVI Mental Health Linkages

        Note:
            Grants: ICJIA - CVI Mental Health Linkages
        '''

        query = f'''
        select count(distinct case when linked_date between {self.q_t1} and {self.q_t2} and client_initiated = 'No' then participant_id else null end) linkages_made,
        count(distinct case when start_date between {self.q_t1} and {self.q_t2} then participant_id else null end) linkages_started
        from neon.linkages
        where (internal_program = 'therapy' or linkage_type = 'mental health') and participant_id in (select participant_id from grant_projections.cvi_full)
        '''

        df = self.query_run(query)
        return df
    
    def cvi_post_incident(self):
        '''
        Returns number of people received services after homicide shooting

        Example:
            Get # of people who received services after homicide shooting::

                e.cvi_post_incident()
        
        Note:
            Grants: ICJIA - CVI Recieved Services after Shooting
        '''
        query = f'''
        select type_incident, sum(num_individuals) as total_ppl from neon.critical_incidents
        where num_deceased > 0 and incident_date between {self.q_t1} and {self.q_t2}
        group by type_incident
        '''
        df = self.query_run(query)
        return(df)


    def idhs_enrollment(self):
        '''
        Returns a table of enrollment numbers for IDHS

        Example:
            Get clients enrolled in IDHS and its services::

                e.idhs_enrollment()
        
        Note:
            Grants: IDHS - VP Enrollment
        '''

        query = f'''select 'TOTAL' as 'clients', count(distinct case when new_client = 'new' then participant_id else null end) as new,
        count(distinct case when new_client = 'continuing' then participant_id else null end) as continuing,
        count(distinct case when discharged_client = 'program' then participant_id else null end) as discharged
        from {self.table}
        UNION ALL
        select service_type, count(distinct case when new_client = 'new' then participant_id 
        when grant_start between {self.q_t1} and {self.q_t2} then participant_id else null end) as new,
        count(distinct case when new_client = 'continuing' and grant_start not between {self.q_t1} and {self.q_t2} then participant_id else null end) as continuing,
        count(distinct case when discharged_client is not null then participant_id else null end) as discharged
        from {self.table}
        group by service_type'''
        df = self.query_run(query)
        return(df)
    
    def idhs_race_gender(self, race_gender = 'race'):
        '''
        Returns a table of client races or genders broken out by new client status

        Parameters:
            race_gender (str): whether to tally 'race' or 'gender'. Defaults to 'race'
        
        Examples:
            See IDHS client races::

                e.idhs_race_gender('race')

            See IDHS client genders::
                
                e.idhs_race_gender('gender')
        
        Note:
            Grants: IDHS - VP Client Races or Genders
        '''

        query = f'''
        select {race_gender}, count(distinct case when new_client = 'new' then participant_id else null end) as new,
        count(distinct case when new_client = 'continuing' then participant_id else null end) as continuing
        from {self.table}
        group by {race_gender}
        '''
        df = self.query_run(query)
        return(df)

    def idhs_language(self):
        '''
        Counts participant primary languages

        Example:

            Get IDHS client primary languages::

                e.idhs_language()
        
        Note:
            Grants: IDHS - VP Client Languages
        '''
        query = f'''select new_client, language_primary, count(distinct participant_id)
        from {self.table}
        group by new_client, language_primary'''
        df = self.query_run(query)
        return df

    def idhs_age(self, cvi = False):
        '''
        Returns a table of client ages broken out by new client status

        Parameters:
            cvi (Bool): whether to use the age groups on the CVI form. Defaults to False
        
        Examples:
            Get IDHS client ages for the CVI::

                e.idhs_age(True)
            
            Get IDHS client ages for the PPR::

                e.idhs_age(False)
        
        Note:
            Grants: IDHS - VP Client Ages
        '''
        if cvi:
            query = f'''
            select 'Under 18' as age_range, count(distinct case when new_client = 'new' then participant_id else null end) as new,
            count(distinct case when new_client = 'continuing' then participant_id else null end) as continuing
            from {self.table}
            where TIMESTAMPDIFF(YEAR, birth_date, grant_start) <18
            UNION ALL
            select '18-24' as age_range, count(distinct case when new_client = 'new' then participant_id else null end) as new,
            count(distinct case when new_client = 'continuing' then participant_id else null end) as continuing
            from {self.table}
            where TIMESTAMPDIFF(YEAR, birth_date, grant_start) BETWEEN 18 AND 24
            UNION ALL
            select '25+' as age_range, count(distinct case when new_client = 'new' then participant_id else null end) as new,
            count(distinct case when new_client = 'continuing' then participant_id else null end) as continuing
            from {self.table}
            where TIMESTAMPDIFF(YEAR, birth_date, grant_start) > 24
            UNION ALL 
            select 'MISSING' as age_range, count(distinct case when new_client = 'new' then participant_id else null end) as new,
            count(distinct case when new_client = 'continuing' then participant_id else null end) as continuing
            from {self.table}
            where birth_date is null'''
        else:
            query = f'''
            select '11-13' as age_range, count(distinct case when new_client = 'new' then participant_id else null end) as new,
            count(distinct case when new_client = 'continuing' then participant_id else null end) as continuing
            from {self.table}
            where TIMESTAMPDIFF(YEAR, birth_date, grant_start) BETWEEN 11 AND 13
            UNION ALL
            select '14-17' as age_range, count(distinct case when new_client = 'new' then participant_id else null end) as new,
            count(distinct case when new_client = 'continuing' then participant_id else null end) as continuing
            from {self.table}
            where TIMESTAMPDIFF(YEAR, birth_date, grant_start) BETWEEN 14 AND 17
            UNION ALL
            select '18-24' as age_range, count(distinct case when new_client = 'new' then participant_id else null end) as new,
            count(distinct case when new_client = 'continuing' then participant_id else null end) as continuing
            from {self.table}
            where TIMESTAMPDIFF(YEAR, birth_date, grant_start) BETWEEN 18 AND 24
            UNION ALL
            select '25+' as age_range, count(distinct case when new_client = 'new' then participant_id else null end) as new,
            count(distinct case when new_client = 'continuing' then participant_id else null end) as continuing
            from {self.table}
            where TIMESTAMPDIFF(YEAR, birth_date, grant_start) > 24
            UNION ALL 
            select 'MISSING' as age_range, count(distinct case when new_client = 'new' then participant_id else null end) as new,
            count(distinct case when new_client = 'continuing' then participant_id else null end) as continuing
            from {self.table}
            where birth_date is null
            '''
        df = self.query_run(query)
        return(df)
    
    
    def idhs_linkages(self, internal_external = False, cm_only = True):
        '''
        Returns a table of linkage information for the quarter
        
        Parameters:
            internal_external: whether to group by internal/external functions. Defaults to False
            cm_only: whether to only count clients with case management funded by grant. Defaults to True
        
        Examples:
            Get IDHS client linkages by linkage category::

                e.idhs_linkages()
            
            Get a breakdown of internal/external linkages for IDHS clients::

                e.idhs_linkages(True)
        
        Note:
            Grants: IDHS - VP Linkage Table
        
        '''
        
        
        int_ext = 'internal_external' if internal_external else 'linkage_type'
        cm_only_statement = "where service_type = 'case management'" if cm_only else ''
        query = f'''
        with link as(
        select participant_id, new_client, case when linkage_type is null then internal_program else linkage_type end as linkage_type, internal_external, linkage_org from neon.linkages
        join (select distinct participant_id, new_client from {self.table} {cm_only_statement}) i using(participant_id)
        where client_initiated = 'no' and linked_date between {self.q_t1} and {self.q_t2})

        select {int_ext}, count(case when new_client = 'new' then participant_id else null end) as new_links,
        count(case when new_client = 'continuing' then participant_id else null end) as cont_links
        from link
        group by {int_ext}
        '''
        df = self.query_run(query)
        return(df)

    def idhs_linkages_detailed(self):
        '''
        Returns a Frankensteined table of other forms of 'detail-level services'. 
        Currently includes in-kind services, outreach/cm assessments, and topics of cm sessions.
        
        Example:
            Get a table of non-linkages services IDHS clients were connected to::

                e.idhs_linkages_detailed()

        Note:
            Grants: IDHS - VP Extended Linkage Table
        '''
        def in_kind_services():
            ### FIX FIX FIX
            query = f'''
            select concat('Received ', service_type) detail_service_type, 
            count(distinct case when new_client = 'new' then participant_id else null end) as new,
            count(distinct case when new_client = 'continuing' then participant_id else null end) as continuing
            from stints.neon
            join (select distinct participant_id, new_client from participants.idhs) i using(participant_id)
            where grant_type not like "IDHS VP"
            group by service_type'''
            df = self.query_run(query)
            return(df)

        def assessments_plus():
            query = f'''
            with part as (select distinct participant_id, new_client from participants.idhs),

            isp_updates as(select 'Received an ISP update' detail_service_type, count(distinct case when new_client = 'new' then participant_id else null end) as new,
            count(distinct case when new_client = 'continuing' then participant_id else null end) as continuing from (select * from isp_tracker
            join part using(participant_id)
            where latest_update between {self.q_t1} and {self.q_t2}) isp),

            cm_assess as(
            select 'Got a cm assessment' count, count(distinct case when new_client = 'new' then participant_id else null end) as new_assess,
            count(distinct case when new_client = 'continuing' then participant_id else null end) as cont_assess
            from(
            select participant_id, new_client, count(score) as num_cm_assess from assess.cm_long
            join part using(participant_id)
            where assessment_date between {self.q_t1} and {self.q_t2}
            group by participant_id, new_client) i),

            ow_elig as (select 'OW elig screening' count, count(distinct case when new_client = 'new' then participant_id else null end) as new_assess,
            count(distinct case when new_client = 'continuing' then participant_id else null end) as cont_assess from (select * from assess.outreach_eligibility
            right join part using(participant_id)
            where assessment_date between {self.q_t1} and {self.q_t2})o),

            ow_assess as (select 'Got an ow assessment' count, count(distinct case when new_client = 'new' then participant_id else null end) as new_assess,
            count(distinct case when new_client = 'continuing' then participant_id else null end) as cont_assess from (select * from assess.outreach_tracker
            right join part using(participant_id)
            where (latest_assessment between {self.q_t1} and {self.q_t2}) or (latest_intervention between {self.q_t1} and {self.q_t2})) o),

            successful_sessions as (select concat("Successful ", contact_type, " Session") sessions, 
            count(distinct case when new_client = 'new' then participant_id else null end) as new_in_kind,
            count(distinct case when new_client = 'continuing' then participant_id else null end) as cont_in_kind
            from (
            select p.participant_id, new_client, contact_type, session_date, description from neon.case_sessions c
            join (select participant_id, new_client, service_type from participants.idhs) p on p.participant_id = c.participant_id and p.service_type = c.contact_type
            where (session_date between {self.q_t1} and {self.q_t2}) and successful_contact = 'Yes') i
            group by contact_type)

            select * from isp_updates
            union all
            select * from cm_assess
            union all
            select * from ow_elig
            union all
            select * from ow_assess
            union all
            select * from successful_sessions
            '''
            df = self.query_run(query)
            return(df)
        
        def cm_sessions():
            query = f'''

            with sess as(select participant_id, focus_contact, contact_type, new_client, description from neon.case_sessions
            join (select distinct participant_id, new_client from participants.idhs where service_type = 'case management') i using(participant_id)
            where (session_date between {self.q_t1} and {self.q_t2}) and successful_contact = 'Yes' and focus_contact is not null),
            separated as
            (select participant_id, new_client, contact_type, SUBSTRING_INDEX(SUBSTRING_INDEX(focus_contact, ', ', n), ', ', -1) AS separated_focus
            from sess
            JOIN (
                SELECT 1 AS n UNION ALL
                SELECT 2 UNION ALL
                SELECT 3 UNION ALL
                SELECT 4 UNION ALL
                SELECT 5 UNION ALL
                SELECT 6 UNION ALL
                SELECT 7 UNION ALL
                SELECT 8 UNION ALL
                SELECT 9
            ) AS numbers
            ON CHAR_LENGTH(focus_contact) - CHAR_LENGTH(REPLACE(focus_contact, ',', '')) >= n - 1)

            select separated_focus detail_service_type, count(distinct case when new_client = 'new' then participant_id else null end) as new,
            count(distinct case when new_client = 'continuing' then participant_id else null end) as continuing from 
            separated
            group by separated_focus
            '''
            df = self.query_run(query)
            return(df)
        
        sessions = cm_sessions().set_index('detail_service_type')
        assess = assessments_plus().set_index('detail_service_type')
        in_kind = in_kind_services().set_index('detail_service_type')
        df = pd.concat([in_kind, assess, sessions], keys=['In-Kind Services','Other Detail-Level Services','Focus of CM Sessions'])
        df = df.reset_index()
        df = df.rename(columns={'level_0':'data_source'})
        return df


    def idhs_incidents(self, CPIC = True):
        '''
        Returns incident analysis for CPIC/non-CPIC notifications

        Example:
            Get a CPIC notification breakdown for IDHS::

                e.idhs_incidents()
            
            Get a non-CPIC notification breakdown for IDHS::

                e.idhs_incidents(False)
        
        Note:
            Grants: IDHS - VP Incident Tally
        '''
        if CPIC:
            query = f'''SELECT type_incident,
            count(case when num_deceased > 0 then incident_id else null end) as fatal,
            count(case when num_deceased = 0 then incident_id else null end) as non_fatal
            FROM neon.critical_incidents
            where how_hear regexp '.*cpic.*' and incident_date between {self.q_t1} and {self.q_t2}
            group by type_incident'''
        else:
            query = f'''select how_hear, count(incident_id) count from neon.critical_incidents
            where incident_date between {self.q_t1} and {self.q_t2}
            group by how_hear'''
        df = self.query_run(query)
        return df

    def idhs_r_schooling_gender(self):
        '''
        Returns client gender counts broken out by schooling status

        Example:
            Get client genders/school statuses for IDHS - R::

                e.idhs_r_schooling_gender()
        
        Note:
            Grants: IDHS - R Client Genders/School Statuses
        '''

        query = f'''
        with edu as (
        select distinct(participant_id), linkage_org from neon.linkages
        where linkage_type like "education%" and start_date is not null and end_status is null
        and (end_date is null or end_date > {self.q_t2}))
        
        select gender, count(distinct case when linkage_org is not null then participant_id else null end) in_school,
        count(distinct case when linkage_org is null then participant_id else null end) other
        from participants.idhs_r
        left join edu using(participant_id)
        where new_client = 'new'
        group by gender
        '''

        df = self.query_run(query)
        return df

    def idhs_r_age_gender(self):
        '''
        Returns client gender counts broken out by age range

        Example:
            Get client genders/ages for IDHS - R::

                e.idhs_r_age_gender()
        
        Note:
            Grants: IDHS - R Client Ages/Genders
        '''

        query = f'''
                select '13 and Under' as age_range, count(distinct case when gender = 'male' then participant_id else null end) as male,
                count(distinct case when gender = 'female' then participant_id else null end) as female,
                count(distinct case when gender not regexp "female|male" then participant_id else null end) as other,
                count(distinct case when gender is null then participant_id else null end) as missing
                from {self.table}
                where TIMESTAMPDIFF(YEAR, birth_date, grant_start) < 14 and new_client = 'new'
                UNION ALL
                select '14-15' as age_range, count(distinct case when gender = 'male' then participant_id else null end) as new,
                count(distinct case when gender = 'female' then participant_id else null end) as continuing,
                count(distinct case when gender not regexp "female|male" then participant_id else null end) as othr,
                count(distinct case when gender is null then participant_id else null end) as missing
                from {self.table}
                where TIMESTAMPDIFF(YEAR, birth_date, grant_start) BETWEEN 14 AND 15 and new_client = 'new'
                UNION ALL
                select '16-17' as age_range, count(distinct case when gender = 'male' then participant_id else null end) as new,
                count(distinct case when gender = 'female' then participant_id else null end) as continuing,
                count(distinct case when gender not regexp "female|male" then participant_id else null end) as othr,
                count(distinct case when gender is null then participant_id else null end) as missing
                from {self.table}
                where TIMESTAMPDIFF(YEAR, birth_date, grant_start) BETWEEN 16 AND 17 and new_client = 'new'
                UNION ALL
                select '18-20' as age_range, count(distinct case when gender = 'male' then participant_id else null end) as new,
                count(distinct case when gender = 'female' then participant_id else null end) as continuing,
                count(distinct case when gender not regexp "female|male" then participant_id else null end) as othr,
                count(distinct case when gender is null then participant_id else null end) as missing
                from {self.table}
                where TIMESTAMPDIFF(YEAR, birth_date, grant_start) BETWEEN 18 AND 20 and new_client = 'new'
                UNION ALL
                select '21-24' as age_range, count(distinct case when gender = 'male' then participant_id else null end) as new,
                count(distinct case when gender = 'female' then participant_id else null end) as continuing,
                count(distinct case when gender not regexp "female|male" then participant_id else null end) as othr,
                count(distinct case when gender is null then participant_id else null end) as missing
                from {self.table}
                where TIMESTAMPDIFF(YEAR, birth_date, grant_start) BETWEEN 21 AND 24 and new_client = 'new'
                UNION ALL
                select '25+' as age_range, count(distinct case when gender = 'male' then participant_id else null end) as new,
                count(distinct case when gender = 'female' then participant_id else null end) as continuing,
                count(distinct case when gender not regexp "female|male" then participant_id else null end) as othr,
                count(distinct case when gender is null then participant_id else null end) as missing
                from {self.table}
                where TIMESTAMPDIFF(YEAR, birth_date, grant_start) > 24 and new_client = 'new'
                UNION ALL 
                select 'MISSING' as age_range, count(distinct case when gender = 'male' then participant_id else null end) as new,
                count(distinct case when gender = 'female' then participant_id else null end) as continuing,
                count(distinct case when gender not regexp "female|male" then participant_id else null end) as othr,
                count(distinct case when gender is null then participant_id else null end) as missing
                from {self.table}
                where birth_date is null and new_client = 'new'
        '''
        
        df = self.query_run(query)
        return df

    
    def r3_ages(self):
        '''
        Returns client ages for groups 6-11, 12-14, 15-17, 18-25, 26+
        
        Note:
            Grants: ICJIA - R3 Client Ages
        '''

        query = f'''
        select '6-11' as age_range, count(distinct case when new_client = 'new' then participant_id else null end) as new,
        count(distinct case when new_client = 'continuing' then participant_id else null end) as continuing
        from {self.table}
        where TIMESTAMPDIFF(YEAR, birth_date, grant_start) BETWEEN 11 AND 13
        UNION ALL
        select '12-14' as age_range, count(distinct case when new_client = 'new' then participant_id else null end) as new,
        count(distinct case when new_client = 'continuing' then participant_id else null end) as continuing
        from {self.table}
        where TIMESTAMPDIFF(YEAR, birth_date, grant_start) BETWEEN 12 AND 14
        UNION ALL
        select '15-17' as age_range, count(distinct case when new_client = 'new' then participant_id else null end) as new,
        count(distinct case when new_client = 'continuing' then participant_id else null end) as continuing
        from {self.table}
        where TIMESTAMPDIFF(YEAR, birth_date, grant_start) BETWEEN 15 AND 17
        UNION ALL
        select '18-25' as age_range, count(distinct case when new_client = 'new' then participant_id else null end) as new,
        count(distinct case when new_client = 'continuing' then participant_id else null end) as continuing
        from {self.table}
        where TIMESTAMPDIFF(YEAR, birth_date, grant_start) BETWEEN 18 AND 25
        UNION ALL
        select '26+' as age_range, count(distinct case when new_client = 'new' then participant_id else null end) as new,
        count(distinct case when new_client = 'continuing' then participant_id else null end) as continuing
        from {self.table}
        where TIMESTAMPDIFF(YEAR, birth_date, grant_start) > 25
        UNION ALL 
        select 'MISSING' as age_range, count(distinct case when new_client = 'new' then participant_id else null end) as new,
        count(distinct case when new_client = 'continuing' then participant_id else null end) as continuing
        from {self.table}
        where birth_date is null
        '''

        df = self.query_run(query)
        return df
 

    def ryds_ages(self):
        '''
        Returns client ages for groups 0-11, 11-13, 14-17, 18-21, 22+

        Note:
            Grants: IDHS - RYDS Client Ages
        '''

        query = f'''
        select '0-10' as age_group, count(distinct participant_id) count from {self.table}
        where TIMESTAMPDIFF(YEAR, birth_date, grant_start) BETWEEN 0 AND 10 and new_client = 'new'
        union all
        select '11-13' as age_group, count(distinct participant_id) count from {self.table}
        where TIMESTAMPDIFF(YEAR, birth_date, grant_start) BETWEEN 11 AND 13 and new_client = 'new'
        union all
        select '14-17' as age_group, count(distinct participant_id) count from {self.table}
        where TIMESTAMPDIFF(YEAR, birth_date, grant_start) BETWEEN 14 AND 17 and new_client = 'new'
        union all
        select '18-21' as age_group, count(distinct participant_id) count from {self.table}
        where TIMESTAMPDIFF(YEAR, birth_date, grant_start) BETWEEN 18 AND 21 and new_client = 'new'
        union all
        select '22+' as age_group, count(distinct participant_id) count from {self.table}
        where TIMESTAMPDIFF(YEAR, birth_date, grant_start) >21 and new_client = 'new'
        union all
        select 'MISSING' as age_group, count(distinct participant_id) count from {self.table}
        where birth_date is null and new_client = 'new'
        '''

        df = self.query_run(query)
        return df
    
    def ryds_cirriculum(self, summary_table = True):
        '''
        Returns a table of cirriculum completion information for clients, by default grouped in a summary_table.  

        Parameters:
            summary_table: True returns a table with bins matching the PPR report. False returns one row per client, with information on which units are missing.
        Note:
            Grants: IDHS - RYDS Cirriculum Completion
        '''
        
        query = f'''
        with complete_table as 
        (select *, (case when num_sessions = 1 then .5 else 1 end) unit_completion from 
        (select participant_id, unit_number, count(participant_id) num_sessions from 
            (select participant_id, attendance, REGEXP_REPLACE(unit_number, '[^0-9]+', '') unit_number from neon.activities_attendance) AA
        WHERE unit_number REGEXP '[0-9]' and attendance = 'Attended'
        group by participant_id, unit_number) ugh),

        unit_table as (select participant_id, round(sum(unit_completion)/6, 2) pct_complete,
        group_concat(distinct case when unit_completion = 1 then unit_number else null end order by unit_number separator ', ') complete_units,
        group_concat(distinct case when unit_completion = .5 then unit_number else null end order by unit_number separator ', ') partial_units
        from complete_table
        group by participant_id),

        big_table as (select participant_id, name, complete_units, partial_units, missing_units,
        ifnull(pct_complete, 0) pct_complete, 
        case 
                when pct_complete between .01 and .49 then "0%-50%" 
                when pct_complete between .5 and .99 then "50%-99%" 
                when pct_complete = 1 then "100%" 
                else "0%" end completion_group from unit_table
        right join (select distinct participant_id, concat(first_name, " ",left(last_name,1), ".") name, "1, 2, 3, 4, 5, 6" as missing_units
        from {self.table}) r using(participant_id))
        '''
        
        if summary_table:
            addendum = 'select completion_group, count(distinct participant_id) count from big_table group by completion_group'
            query = query + ' ' + addendum
            df = self.query_run(query)
            return df
        
        addendum = 'select * from big_table'
        query = query + ' ' + addendum
        df = self.query_run(query)
        
        copy_df = df.copy()
        copy_df['missing_units'] = copy_df['missing_units'].apply(lambda x: x.split(', ') if pd.notna(x) else [])
        copy_df['partial_units'] = copy_df['partial_units'].apply(lambda x: x.split(', ') if pd.notna(x) else [])
        copy_df['complete_units'] = copy_df['complete_units'].apply(lambda x: x.split(', ') if pd.notna(x) else [])

        # remove units from the missing units list that appear in the other two
        df['missing_units'] = copy_df.apply(
            lambda row: [
                elem for elem in row['missing_units']
                if elem not in set(row['complete_units'] + row['partial_units'])
            ],
            axis=1)
        # revert to string
        df['missing_units'] = df['missing_units'].transform(lambda x: ','.join(map(str, x)))
        return df