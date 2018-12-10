from django.db import models
import django.utils.timezone as timezone


class Result(models.Model):
    log_filename = models.CharField(max_length=250, default='con050217.csv')
    start_time = models.CharField(max_length=100, default='2018-05-02T17:00')
    end_time = models.CharField(max_length=100, default='2018-05-02T17:01')
    smallest_size = models.IntegerField(default=2)
    interval = models.IntegerField(default=60)
    ordinal_number = models.IntegerField(default=1)
    formatted_args = models.CharField(max_length=100, default="")
    link_counts = models.IntegerField(default=0)
    community_counts = models.IntegerField(default=0)
    ip_counts = models.IntegerField(default=0)
    result_time = models.DateTimeField(default=timezone.now())
    result_filename = models.CharField(primary_key=True, max_length=200,
                                       default='con050217.csv_2018-05-02T1700_2018-05-02T1701_2.csv')

    def __str__(self):
        return str(self.result_filename) + ' -- ' + str(self.result_time)


class Community(models.Model):
    result = models.ForeignKey(Result, on_delete=models.CASCADE)
    community_tag = models.IntegerField(primary_key=True, default=-1)
    leader_ip = models.CharField(max_length=20, default='')
    ip_counts = models.IntegerField(default=0)
    link_counts = models.IntegerField(default=0)
    purity = models.FloatField(default=0)

    def __str__(self):
        return '{},{},{},{}'.format(self.community_tag, self.leader_ip, self.ip_counts, self.link_counts)
