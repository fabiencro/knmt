import sqlite3
import bokeh.charts
def commandline():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("sqldb")
    args = parser.parse_args()
    
    db = sqlite3.connect(args.sqldb)
    c = db.cursor()
    
    c.execute("SELECT * FROM exp_data")
    all = [[], [], [], []]
    for line in c.fetchall():
        print line
        _, _, _, test_loss, test_bleu, dev_loss, dev_bleu = line
        all[0].append(test_loss)
        all[1].append(test_bleu)
        all[2].append(dev_loss)
        all[3].append(dev_bleu)
    bokeh.charts.output_file("graph_loss.html")
    
    p = bokeh.charts.Line(all)
    bokeh.charts.show(p)
    
if __name__ == '__main__':
    commandline()