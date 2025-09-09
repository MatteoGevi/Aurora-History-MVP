import requests, fitz
url="https://hyktlmgolbfnqwaptkqy.supabase.co/storage/v1/object/sign/textbook/ai-engineering-building-applications-with-foundation-models.pdf?token=eyJraWQiOiJzdG9yYWdlLXVybC1zaWduaW5nLWtleV82OTc2NTQ2Mi05OWYzLTQyODMtYTRiZi03ZjI3MjhlMTEyZTMiLCJhbGciOiJIUzI1NiJ9.eyJ1cmwiOiJ0ZXh0Ym9vay9haS1lbmdpbmVlcmluZy1idWlsZGluZy1hcHBsaWNhdGlvbnMtd2l0aC1mb3VuZGF0aW9uLW1vZGVscy5wZGYiLCJpYXQiOjE3NTczNDMyMjUsImV4cCI6MTc4ODg3OTIyNX0.d8Y415Z-dKpfF8apTFewD1u7_dNg43Umn3X9YE3RZ58"

pdf_bytes = requests.get(url).content
doc = fitz.open(stream=pdf_bytes, filetype="pdf")

print("Pages:", doc.page_count)
print("First page text:", doc[0].get_text("text")[:300])