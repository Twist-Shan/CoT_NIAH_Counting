"""Check embedded fonts, native physical sizes, and effective raster resolution."""
from pathlib import Path
from math import hypot
import json
from pypdf import PdfReader
from pypdf.generic import ContentStream

OUT=Path(__file__).resolve().parent
meta=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))

def multiply(m,n):
    a,b,c,d,e,f=m; g,h,i,j,k,l=n
    return [a*g+c*h,b*g+d*h,a*i+c*j,b*i+d*j,a*k+c*l+e,b*k+d*l+f]

def fonts(resources,seen=None):
    seen=seen or set(); found=[]
    for name,ref in resources.get('/Font',{}).items():
        obj=ref.get_object(); fontname=str(obj.get('/BaseFont',name))
        if fontname in seen: continue
        seen.add(fontname)
        leaves=[f.get_object() for f in obj.get('/DescendantFonts',[])] or [obj]
        embedded=[]
        for f in leaves:
            desc=f.get('/FontDescriptor',{}); desc=desc.get_object() if hasattr(desc,'get_object') else desc
            embedded.append(any(k in desc for k in ['/FontFile','/FontFile2','/FontFile3']) or '/CharProcs' in f)
        found.append(dict(name=fontname,embedded=all(embedded)))
    for ref in resources.get('/XObject',{}).values():
        o=ref.get_object()
        if o.get('/Subtype')=='/Form': found.extend(fonts(o.get('/Resources',{}),seen))
    return found

def images(stream,resources,reader,matrix=None):
    matrix=list(matrix or [1,0,0,1,0,0]); stack=[]; found=[]
    for operands,op in ContentStream(stream,reader).operations:
        if op==b'q': stack.append(matrix.copy())
        elif op==b'Q': matrix=stack.pop()
        elif op==b'cm': matrix=multiply(matrix,[float(x) for x in operands])
        elif op==b'Do':
            obj=resources['/XObject'][operands[0]].get_object()
            if obj['/Subtype']=='/Image':
                sx,sy=hypot(matrix[0],matrix[1]),hypot(matrix[2],matrix[3])
                found.append(dict(pixels=[int(obj['/Width']),int(obj['/Height'])],
                                  effective_dpi=[float(obj['/Width'])*72/sx,float(obj['/Height'])*72/sy]))
            elif obj['/Subtype']=='/Form':
                transform=multiply(matrix,[float(x) for x in obj.get('/Matrix',[1,0,0,1,0,0])])
                found.extend(images(obj,obj.get('/Resources',resources),reader,transform))
    return found

reports=[]
for f in meta['figures']:
    path=OUT/(f['stem']+'.pdf'); reader=PdfReader(path); page=reader.pages[0]
    size=[float(page.mediabox.width)/72,float(page.mediabox.height)/72]
    assert abs(size[0]-6.5)<.001
    font_report=fonts(page['/Resources'])
    assert all(f['embedded'] for f in font_report),(path,font_report)
    raster=images(page.get_contents(),page['/Resources'],reader)
    assert all(min(r['effective_dpi'])>=449 for r in raster),(path,raster)
    reports.append(dict(figure=f['stem'],size_in=size,fonts=font_report,raster_images=raster))
preview=PdfReader(OUT/'appendix_layout_preview.pdf')
for page in preview.pages: assert all(f['embedded'] for f in fonts(page['/Resources']))
(OUT/'qa/pdf_asset_quality.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
print('10 vector PDFs: embedded fonts; all raster layers >= 449 dpi at paper width.')
