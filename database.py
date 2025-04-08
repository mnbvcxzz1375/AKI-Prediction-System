import psycopg2
#建立数据库连接
con = psycopg2.connect(database="mimiciii",
                       user="postgres",
                       password="123456",
                       host="localhost",
                       port="5433")
#调用游标对象
cur = con.cursor()


# cur.execute("SELECT row_id,subject_id,hadm_id,icustay_id,itemid,charttime,storetime,cgid,value,valuenum,valueuom,warning,error,resultstatus,stopped from chartevents_1")
# rows = cur.fetchall()

# for row in rows:
#     print("row_id =", row[0])
#     print("subject_id =", row[1])
#     print("hadm_id =", row[2])
#     print("icustay_id =", row[3])
#     print("itemid =", row[4])
#     print("charttime =", row[5])
#     print("stroetime =", row[6])
#     print("cgid =", row[7])
#     print("value =", row[8])
#     print("valuenum =", row[9])
#     print("valueom =", row[10])
#     print("warning =", row[11])
#     print("error =", row[12])
#     print("resultstatus =", row[13])
#     print("stopped =", row[14], "\n")
#
# print("Operation done successfully")


cur.execute('''SELECT distinct d.row_id, d.subject_id, d.hadm_id, d.seq_num, d.icd9_code
                FROM diagnoses_icd d join labevents lab WHERE icd9_code LIKE '584%' or icd9_code like '586%' ;''')
rows = cur.fetchall()

i=0
for row in rows:
    i=i+1
    print(i)
    print("row_id =", row[0])
    print("subject_id =", row[1])
    print("hadm_id =", row[2])
    print("seq_num =", row[3])
    print("icd9_code =", row[4],"\n")

con.close()
