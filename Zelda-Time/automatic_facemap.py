import os
import facemap
import datetime
import time
import batch_process_pinkavrig_videos as bprocess


def main():
    how_often_to_check = 3600
    continue_running = True

    while continue_running:
        e = datetime.datetime.now()
        print("The time is now: %s:%s:%s" % (e.hour, e.minute, e.second))

        hour_str = '%s' % e.hour
        hour_int = int(hour_str)

        if (hour_int < 8) | (hour_int >= 20):
            print('It is prime time to run some facemap!')
            bprocess.main()

        else:
            print('Hi Magda, it is between 8am - 8pm, I will sleep a bit and check back later')
            time.sleep(how_often_to_check)

if __name__ == '__main__':
    main()