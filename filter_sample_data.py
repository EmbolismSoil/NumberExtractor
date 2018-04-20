from optparse import OptionParser

def filter(f, o):
    records = {}
    for line in f.readlines():
        line = line.strip('\n')
        items = line.split('\t')
        if len(items) != 2:
            continue
        records[items[1]] = items[0]

    for qq, sms in records.items():
        line = '%s\t%s\n' % (sms, qq)
        o.write(line)

    o.flush()


if __name__ == '__main__':
    parser = OptionParser(usage='%prog -f <raw_data_file> -o <filtered_data_file>')
    parser.add_option('-f', dest='f', help='原始数据文件', metavar='FILE')
    parser.add_option('-o', dest='o', help='过滤后的输出文件', metavar='FILE')

    args, opts = parser.parse_args()

    with open(args.f, 'r') as f, open(args.o, 'w+') as o:
        filter(f, o)