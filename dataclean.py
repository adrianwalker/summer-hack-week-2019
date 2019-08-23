import csv

if __name__ == "__main__":
    headers = None
    cleaned_rows = []
    cleaned_rows2 = []
    with open("pima-indians-diabetes.csv", "r") as datafile:
        reader = csv.reader(datafile)

        row_count = 0

        for index, row in enumerate(reader):
            if index == 0:
                headers = row
            else:
                row_count += 1
                if int(row[1]) is not 0:
                    cleaned_rows.append(row)

        # print(cleaned_rows[1][2])
        for row in cleaned_rows:
            if int(row[2]) is not 0:
                cleaned_rows2.append(row)

        cleaned_rows = []
        for row in cleaned_rows2:
            if float(row[5]) != 0.0:
                cleaned_rows.append(row)

        print(row_count)
        print(len(cleaned_rows))

        # print(cleaned_rows)
        with open("cleaned-data.csv", "w+") as writefile:
            writer = csv.writer(writefile)
            writer.writerow(headers)
            writer.writerows(cleaned_rows)
            writefile.close()
            # print(dir(writer))
