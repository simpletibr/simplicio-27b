def log_message(filename: str, msg: str):
    f = open(filename, 'a')
    f.write(msg + '\n')
