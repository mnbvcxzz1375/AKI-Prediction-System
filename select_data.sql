
-- 血气
select distinct bl.subject_id,bl.hadm_id,d.icd9_code,bl.so2,bl.spo2,bl.po2,bl.pco2,bl.fio2,bl.aado2,bl.pao2fio2,
	bl.ph,bl.totalco2,bl.temperature,p.gender,bl.requiredo2,bl.tidalvolume,bl.calcium
from blood_gas_first_day_arterial bl
join diagnoses_icd d on d.hadm_id=bl.hadm_id
join d_icd_diagnoses di on di.icd9_code=d.icd9_code
join patients p  on p.subject_id = d.subject_id
where di.long_title ilike '% renal fail%' 
	or di.long_title ilike '%kidney fail%'
	or di.long_title ilike '%liver fail%'
	or di.long_title ilike '%spleen fail%'
	
	
select distinct icd9_code from d_icd_diagnoses where  long_title ilike '% renal fail%' 
	or long_title ilike '%kidney fail%'
	or long_title ilike '%liver fail%'
	or long_title ilike '%spleen fail%'

select * from kdigo_creatinine
select * from kdigo_stages_48hr

--aki患者肌酐水平+第一天实验室数据
select distinct * from labs_first_day lf
join kdigo_creatinine kc on kc.icustay_id=lf.icustay_id
join diagnoses_icd d on d.hadm_id=lf.hadm_id
join d_icd_diagnoses di on di.icd9_code=d.icd9_code
join patients p  on p.subject_id = d.subject_id
where (di.long_title ilike '% renal fail%' 
	or di.long_title ilike '%kidney fail%'
	or di.long_title ilike '%liver fail%'
	or di.long_title ilike '%spleen fail%')
	
--aki患者肌酐水平+微生物实验室数据	
select distinct * from microbiologyevents mb
join icustay_detail icd on  icd.hadm_id = mb.hadm_id
join kdigo_creatinine kc on kc.icustay_id=icd.icustay_id
join diagnoses_icd d on d.hadm_id=mb.hadm_id
join d_icd_diagnoses di on di.icd9_code=d.icd9_code
join patients p  on p.subject_id = d.subject_id
where (di.long_title ilike '% renal fail%' 
	or di.long_title ilike '%kidney fail%'
	or di.long_title ilike '%liver fail%'
	or di.long_title ilike '%spleen fail%')
	

--aki肾衰竭患者尿量数据+第一天实验室数据
select distinct * from labs_first_day lf
join kdigo_uo ku on ku.icustay_id=lf.icustay_id
join diagnoses_icd d on d.hadm_id=lf.hadm_id
join d_icd_diagnoses di on di.icd9_code=d.icd9_code
join patients p  on p.subject_id = d.subject_id
where (di.long_title ilike '% renal fail%' 
	or di.long_title ilike '%kidney fail%'
	or di.long_title ilike '%liver fail%'
	or di.long_title ilike '%spleen fail%')
	
--住院第一天的生命体征与aki肾衰竭患者尿量数据	
select distinct* from vitals_first_day vfd
join kdigo_uo ku on ku.icustay_id = vfd.icustay_id
join diagnoses_icd d on d.hadm_id=vfd.hadm_id
join d_icd_diagnoses di on di.icd9_code=d.icd9_code
join patients p  on p.subject_id = d.subject_id
where (di.long_title ilike '% renal fail%' 
	or di.long_title ilike '%kidney fail%'
	or di.long_title ilike '%liver fail%'
	or di.long_title ilike '%spleen fail%')