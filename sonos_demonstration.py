import soco

VOLUME = 70
SAMPLE_MP3 = "https://s3.amazonaws.com/theme_music/The+Lion+King+-+Circle+of+Life.mp3"


def find_sonos_device(target_speaker_name):
    sonos_devices = soco.SonosDiscovery()
    print "Searching for Sonos device..."
    while True:
        for ip in sonos_devices.get_speaker_ips():
            device = soco.SoCo(ip)
            speaker_name = device.get_speaker_info().get("zone_name")
            if speaker_name == target_speaker_name:
                print "Found %s" % target_speaker_name
                return device
            else:
                print "Found %s, but not %s" % (speaker_name, target_speaker_name)
        print "Found nothing.  Retrying..."


if __name__ == "__main__":
    device = find_sonos_device("Recruiting")
    print "Playing %s" % SAMPLE_MP3
    device.volume(VOLUME)
    device.play_uri(SAMPLE_MP3)
    print "Finished."
