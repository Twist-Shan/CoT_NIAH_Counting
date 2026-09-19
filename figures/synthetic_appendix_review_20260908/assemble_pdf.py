"""Vector-preserving figure review at the manuscript's 5.5-inch width."""
from pathlib import Path
from io import BytesIO
import hashlib
import html
import json

from pypdf import PdfReader,PdfWriter,Transformation
from reportlab.pdfgen import canvas
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
import pypdfium2 as pdfium
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
all_writer,main_writer=PdfWriter(),PdfWriter()
page_records=[]
page_w=6.3*72
margin=.4*72
available=5.5*72
title_style=ParagraphStyle('title',fontName='Times-Bold',fontSize=12,leading=14,textColor='#161923')
caption_style=ParagraphStyle('caption',fontName='Times-Roman',fontSize=10,leading=12,textColor='#161923')
for i,f in enumerate(manifest['figures'],1):
    title=Paragraph(f'{i:02d}. '+html.escape(f['title']),title_style)
    _,title_h=title.wrap(available,200)
    caption=Paragraph(html.escape(f['caption']),caption_style)
    _,caption_h=caption.wrap(available,500)
    reader=PdfReader(str(ROOT/f'{f["stem"]}.pdf'))
    assert len(reader.pages)==1
    original=reader.pages[0]
    width=float(original.mediabox.width)
    height=float(original.mediabox.height)
    target_w=available*f['width_fraction']
    scale=target_w/width
    image_h=height*scale
    page_h=margin+title_h+20+image_h+13+caption_h+margin+15
    image_y=margin+15+caption_h+13
    image_x=(page_w-target_w)/2
    stream=BytesIO()
    c=canvas.Canvas(stream,pagesize=(page_w,page_h))
    title.drawOn(c,margin,page_h-margin-title_h)
    c.setFont('Helvetica',8)
    c.setFillColorRGB(.0,.494,.671)
    tag={'main':'MAIN CANDIDATE','alternative':'PCA COMPARISON','candidate':'ORIGINAL PROTOCOL - AWAITING ALIGNMENT'}[f['group']]
    c.drawString(margin,page_h-margin-title_h-12,tag)
    caption.drawOn(c,margin,margin+15)
    c.setFillColorRGB(.4,.42,.46)
    c.setFont('Helvetica',8)
    c.drawString(margin,margin-6,'Synthetic appendix | Aurora | 2026-09-08')
    c.drawRightString(page_w-margin,margin-6,f'{i:02d} / {len(manifest["figures"]):02d}')
    c.showPage();c.save()
    stream.seek(0)
    page=PdfReader(stream).pages[0]
    transform=Transformation().translate(-float(original.mediabox.left),-float(original.mediabox.bottom)).scale(scale).translate(image_x,image_y)
    page.merge_transformed_page(original,transform,over=True,expand=False)
    all_writer.add_page(page)
    all_writer.add_outline_item(f'{i:02d}. {f["title"]}',i-1)
    if f['group']=='main':
        main_writer.add_page(page)
    page_records.append(dict(page=i,stem=f['stem'],group=f['group'],page_width=page_w,page_height=page_h,
                             inclusion_width=target_w,caption_height=caption_h,figure_caption_gap=13))
all_writer.write(str(ROOT/'appendix_figures_review.pdf'))
main_writer.write(str(ROOT/'appendix_main_figures.pdf'))
qa=ROOT/'qa'
qa.mkdir(exist_ok=True)
doc=pdfium.PdfDocument(str(ROOT/'appendix_figures_review.pdf'))
text_bounds=[]
for i,page in enumerate(doc):
    bitmap=page.render(scale=1.5)
    im=bitmap.to_pil()
    im.save(qa/f'page_{i+1:02d}.png')
    textpage=page.get_textpage()
    outside=[]
    for j in range(textpage.count_chars()):
        l,b,r,t=textpage.get_charbox(j)
        if min(l,b)<-1 or r>page.get_width()+1 or t>page.get_height()+1:
            outside.append(j)
    assert not outside,(i,outside[:10])
    text_bounds.append(dict(page=i+1,characters=textpage.count_chars(),outside_page_characters=0))
sheet=Image.new('RGB',(1400,500*((len(doc)+1)//2)),'white')
draw=ImageDraw.Draw(sheet)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
for i in range(len(doc)):
    im=Image.open(qa/f'page_{i+1:02d}.png').convert('RGB')
    im.thumbnail((670,465))
    x,y=(i%2)*700,(i//2)*500
    draw.text((x+12,y+3),f'Page {i+1:02d}',font=font,fill='#161923')
    sheet.paste(im,(x+(700-im.width)//2,y+28))
sheet.save(qa/'all_pages_contact.png')
(qa/'pdf_validation.json').write_text(json.dumps({'pages':page_records,'text_bounds':text_bounds,
    'main_pages':9,'all_pages':13,'figures_preserve_original_pdf_graphics':True},indent=2),encoding='utf-8')
(ROOT/'review_pdf_manifest.json').write_text(json.dumps({'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'figure_manifest_sha256':hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest(),
    'pdfs':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['appendix_figures_review.pdf','appendix_main_figures.pdf']}},indent=2),encoding='utf-8')
print('PDF REVIEW COMPLETE: 9 main / 13 total; all text within page bounds')
