from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
import argparse
import os
import subprocess
import io


def write_track_from_notes(track, notes, channel=0, program=None, track_name=None):
    """Convert absolute-time notes to delta-time MIDI messages and append to track.

    notes: iterable of dicts with keys: note(int), start(int ticks), duration(int ticks), velocity(int, optional)
    channel: MIDI channel
    program: program number to set at start (optional)
    track_name: string to add as MetaMessage
    """
    if track_name:
        track.append(MetaMessage('track_name', name=track_name, time=0))

    if program is not None:
        track.append(Message('program_change', program=program, time=0, channel=channel))

    events = []  # list of (abs_time, Message)
    for n in notes:
        vel = n.get('velocity', 64)
        start = int(n['start'])
        end = int(n['start'] + n['duration'])
        events.append((start, Message('note_on', note=int(n['note']), velocity=vel, time=0, channel=channel)))
        events.append((end, Message('note_off', note=int(n['note']), velocity=vel, time=0, channel=channel)))

    # sort by absolute time, stable so note_on before note_off at same time if created that way
    events.sort(key=lambda x: x[0])

    last_time = 0
    for abs_time, msg in events:
        delta = abs_time - last_time
        if delta < 0:
            delta = 0
        msg.time = delta
        track.append(msg)
        last_time = abs_time


def generate_example_midi(outname='output.mid', bpm=120, ticks_per_beat=480):
    """Generate a simple multi-track MIDI and save to outname.
    Returns the path to the saved MIDI file.
    """
    mid = MidiFile(type=1, ticks_per_beat=ticks_per_beat)

    # Tempo track (track 0)
    tempo_track = MidiTrack()
    tempo_track.append(MetaMessage('track_name', name='Tempo', time=0))
    tempo_track.append(MetaMessage('set_tempo', tempo=bpm2tempo(bpm), time=0))
    mid.tracks.append(tempo_track)

    # Create instrument tracks
    piano = MidiTrack()
    strings = MidiTrack()
    choir = MidiTrack()
    guitar = MidiTrack()
    bass = MidiTrack()
    drums = MidiTrack()

    mid.tracks.extend([piano, strings, choir, guitar, bass, drums])

    # Define notes as absolute times (ticks)
    piano_notes = [
        {'note': 60, 'start': 0, 'duration': 480, 'velocity': 90},
        {'note': 64, 'start': 0, 'duration': 480, 'velocity': 90},
    ]

    strings_notes = [
        {'note': 67, 'start': 0, 'duration': 960, 'velocity': 70},
    ]

    choir_notes = [
        {'note': 72, 'start': 0, 'duration': 960, 'velocity': 70},
    ]

    guitar_notes = [
        {'note': 60, 'start': 0, 'duration': 240, 'velocity': 80},
        {'note': 64, 'start': 240, 'duration': 240, 'velocity': 80},
        {'note': 67, 'start': 240, 'duration': 480, 'velocity': 80},
    ]

    bass_notes = [
        {'note': 36, 'start': 0, 'duration': 960, 'velocity': 100},
    ]

    drums_notes = [
        {'note': 38, 'start': 0, 'duration': 120, 'velocity': 100},  # Snare
        {'note': 42, 'start': 120, 'duration': 120, 'velocity': 80},  # Hi-hat
    ]

    # Write tracks
    write_track_from_notes(piano, piano_notes, channel=0, program=0, track_name='Piano')
    write_track_from_notes(strings, strings_notes, channel=1, program=48, track_name='Strings')
    write_track_from_notes(choir, choir_notes, channel=2, program=52, track_name='Choir')
    write_track_from_notes(guitar, guitar_notes, channel=3, program=24, track_name='Guitar')
    write_track_from_notes(bass, bass_notes, channel=4, program=32, track_name='Bass')
    write_track_from_notes(drums, drums_notes, channel=9, program=None, track_name='Drums')

    mid.save(outname)
    return outname


def render_wav_with_fluidsynth(midfile, soundfont, outwav, sample_rate=44100):
    """Render a MIDI file to WAV using external fluidsynth command.

    Requires `fluidsynth` in PATH and a valid SoundFont file (.sf2).
    """
    if not os.path.isfile(midfile):
        raise FileNotFoundError(f'MIDI file not found: {midfile}')
    if not os.path.isfile(soundfont):
        raise FileNotFoundError(f'SoundFont .sf2 not found: {soundfont}')

    fluidsynth_cmd = [
        'fluidsynth', '-ni', soundfont, midfile,
        '-F', outwav, '-r', str(sample_rate)
    ]
    try:
        subprocess.run(fluidsynth_cmd, check=True)
    except FileNotFoundError:
        raise RuntimeError('fluidsynth not found. Install fluidsynth and ensure it is in your PATH.')

    return outwav


def parse_args_and_run():
    parser = argparse.ArgumentParser(description='Generate example MIDI and optionally render to WAV with fluidsynth')
    parser.add_argument('--tempo', type=int, default=120, help='Tempo in BPM')
    parser.add_argument('--out', type=str, default='Antena_do_Seculo_Vindouro_Marilia_fixed.mid', help='Output MIDI filename')
    parser.add_argument('--wav', type=str, default=None, help='If provided, path to output WAV file (requires --soundfont)')
    parser.add_argument('--soundfont', type=str, default=None, help='Path to .sf2 SoundFont used by fluidsynth')
    args = parser.parse_args()

    midpath = generate_example_midi(outname=args.out, bpm=args.tempo)
    print(f'MIDI salvo: {midpath}')

    if args.wav:
        if not args.soundfont:
            parser.error('--wav requires --soundfont to be provided')
        wavpath = render_wav_with_fluidsynth(midpath, args.soundfont, args.wav)
        print(f'WAV renderizado: {wavpath}')


if __name__ == '__main__':
    parse_args_and_run()
