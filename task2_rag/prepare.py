"""Download public corpus and freeze manually authored question/passage labels."""
from pathlib import Path
import requests,json,hashlib,re
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parent
URL='https://www.nasa.gov/wp-content/uploads/static/apollo50th/pdf/A11_MissionReport.pdf'
def prepare():
 p=ROOT/'data'/'apollo11.pdf'; p.parent.mkdir(exist_ok=True)
 if not p.exists():
  r=requests.get(URL,timeout=120); r.raise_for_status(); p.write_bytes(r.content)
 pages=[{'page':i+1,'text':re.sub(r'\s+',' ',p.extract_text()).strip()} for i,p in enumerate(PdfReader(p).pages)]
 (ROOT/'data'/'pages.json').write_text(json.dumps(pages,indent=2))
 # Page numbers are one-based PDF page indices, not printed chapter numbers.
 specs=[
 ('What were the three Apollo 11 crew members and their roles?',12,'The crew were','The space vehicle'),
 ('Which lunar sea did Apollo 11 land in?',12,'The spacecra ft landed','The lan ding'),
 ('Why was the ocean landing point moved by 215 miles?',13,'In\u00ad clement','The entry'),
 ('Which ship recovered the Apollo 11 astronauts?',13,'After landin','The command module'),
 ('What is the reference time used for elapsed times in this report?',14,'In this report','All references'),
 ('Why did the astronauts explore the surface before their planned sleep?',16,'A crew rest period','After the crew'),
 ('When were the crew and spacecraft released from quarantine?',17,'The crew and',''),
 ('What prevented the Commander from sleeping on the lunar surface?',45,'The rest period','4.1 3'),
 ('Why did manual descent consume more reaction-control propellant than automatic descent?',55,'The reaction control propellant','The desc ent'),
 ('How long did the lunar television transmission operate continuously?',160,'Telev isio n','Lunar module voice'),
 ('What was wrong with the lunar module onboard voice recorder playback?',160,'When the tape','For further'),
 ('Why was actual cabin oxygen use below the preflight prediction?',190,'The actual oxygen','In the following'),
 ('What frequency dominated seismic signals made by astronaut activity?',215,'The pre dominant','This fre'),
 ('What was the likely cause of the service propulsion nitrogen leak?',280,'The most likely cause','The injector'),
 ('How much did nitrogen pressure fall during the lunar orbit insertion firing?',280,'During the lunar orbit','The normal')]
 # Exact offsets are selected from visible extraction; store whole relevant paragraphs
 # as annotation context where OCR splits prevent a reliable literal substring.
 questions=[]
 for i,(q,page,a,b) in enumerate(specs,1):
  text=pages[page-1]['text']
  start=text.find(a); end=text.find(b,start+1) if b else len(text)
  if start<0: raise ValueError((i,a))
  if end<0: raise ValueError((i,b))
  passage=text[start:end].strip(); assert passage
  questions.append({'id':i,'question':q,'page':page,'start':start,'end':end,'source_passage':passage})
 (ROOT/'data'/'questions.json').write_text(json.dumps(questions,indent=2))
 (ROOT/'data'/'provenance.json').write_text(json.dumps({'url':URL,'pages':len(pages),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'annotation':'15 questions authored before retrieval; exact source offsets in whitespace-normalized pypdf extraction'},indent=2))
if __name__=='__main__': prepare()
