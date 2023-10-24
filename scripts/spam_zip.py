# Not working
# Don't know how the ZipFile can be used

from zipfile import ZipFile

data_file = 'spam_sms_mini.txt'
zip_file = '../data/spam_sms_mini.zip'
#zf = ZipFile(zip_file, mode='w')
#zf.write(data_file)

with ZipFile(zip_file, mode='r') as myzip:
    with myzip.open(data_file, mode='r') as myfile:
        for s in myfile.readlines():
            processed_str = s.split(sep=" ", maxsplit=1)
            print(processed_str[0], processed_str[1][:30])