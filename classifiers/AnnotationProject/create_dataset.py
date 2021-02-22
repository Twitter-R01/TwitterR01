import sys, csv
'''
Read tsv files of annotated pre and post data and create tsv files of datasets:
D1: complete data with relevant (relevant=1) and non-relevant (relevant=0)
D2: subset of D1 with commercial (com_vape=1) and non-commercial (com_vape=0)
D3: subset of D2 with pro (pro_vape=1), non-pro (pro_vape=0), anti (anti_vape=1) and non-anti (anti_vape=0)

TSV files generated contain all metadata fields, no filtering done in this script.
'''
def get_datasets(reader1, reader2):
	header = reader1.fieldnames
	ofile1 = open('data/D1.tsv', 'w')
	ofile2 = open('data/D2.tsv', 'w')
	ofile3 = open('data/D3.tsv', 'w')
	writer1 = csv.DictWriter(ofile1, fieldnames=header, dialect='excel-tab')
	writer2 = csv.DictWriter(ofile2, fieldnames=header, dialect='excel-tab')
	writer3 = csv.DictWriter(ofile3, fieldnames=header, dialect='excel-tab')

	writer1.writeheader()
	writer2.writeheader()
	writer3.writeheader()

	for row in reader1:
		writer1.writerow(row)
		if row['relevant'] == '1':
			writer2.writerow(row)
			if row['com_vape'] == '0':
				writer3.writerow(row)
	for row in reader2:
		writer1.writerow(row)
		if row['relevant'] == '1':
			writer2.writerow(row)
			if row['com_vape'] == '0':
				writer3.writerow(row)

if __name__ == '__main__':

	file1 = open(sys.argv[1], 'r')
	file2 = open(sys.argv[2], 'r')
	reader1 = csv.DictReader(file1, dialect='excel-tab')
	reader2 = csv.DictReader(file2, dialect='excel-tab')

	if reader1.fieldnames == reader2.fieldnames:
		get_datasets(reader1, reader2)
	else:
		print('Header names do not match')

